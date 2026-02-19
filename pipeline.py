"""
Discovery Pipeline — The Core Orchestration Engine.

This is the brain of the AI Science Discovery Team. It coordinates all 10 steps
of the discovery pipeline, managing data flow between agents, splitting proposals
into individual evaluation tracks, and assembling the final results.

Architecture Summary:
    ┌──────────────┐
    │   FRONTIER    │──► Orchestrator ──► Hypothesis Generator ──┐
    │   PROBLEM     │                                             │
    └──────────────┘                                              │
                                                                  │
         ┌────────────────────────────────────────────────────────┘
         │  FOR EACH HYPOTHESIS:
         │
         ├──► Step Decomposer ──► [decontextualized questions]
         │                              │
         │                    FOR EACH QUESTION:
         │                              │
         │                    ├──► Physics Oracle ──► [validated answer]
         │                              │
         │                    └─────────┘
         │
         ├──► Chain Assembler ──► [validated pathway]
         │
         ├──► Engineering Proposer ──► [engineering proposals]
         │
         ├──► Requirement Challenger ◄──┐ (LOOP x N)
         │         │                    │
         │         └──► Physics Oracle ─┘
         │
         └──► Overseer ──► Final Evaluator ──► DISCOVERY THESIS
"""

import json
import os
import re
import time
from datetime import datetime

from config import (
    AGENT_CONFIGS, NUM_HYPOTHESES, CHALLENGE_ITERATIONS,
    NUM_FINAL_PROPOSALS, RESULTS_DIR,
)
from agents import (
    ORCHESTRATOR_SYSTEM, ORCHESTRATOR_USER_TEMPLATE,
    HYPOTHESIS_GENERATOR_SYSTEM, HYPOTHESIS_GENERATOR_USER_TEMPLATE,
    STEP_DECOMPOSER_SYSTEM, STEP_DECOMPOSER_USER_TEMPLATE,
    PHYSICS_ORACLE_SYSTEM, PHYSICS_ORACLE_USER_TEMPLATE,
    CHAIN_ASSEMBLER_SYSTEM, CHAIN_ASSEMBLER_USER_TEMPLATE,
    ENGINEERING_PROPOSER_SYSTEM, ENGINEERING_PROPOSER_USER_TEMPLATE,
    REQUIREMENT_CHALLENGER_SYSTEM, REQUIREMENT_CHALLENGER_USER_TEMPLATE,
    OVERSEER_SYSTEM, OVERSEER_USER_TEMPLATE,
    FINAL_EVALUATOR_SYSTEM, FINAL_EVALUATOR_USER_TEMPLATE,
)
from llm_client import LLMClient


def safe_json_parse(text):
    """
    Try to parse JSON from model output. Models often wrap JSON in markdown 
    code blocks or add extra text. This tries multiple strategies.
    """
    if not text:
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find first { or [ and last } or ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        end_idx = text.rfind(end_char)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass

    # Strategy 4: Return as plain text wrapped in dict
    return {"raw_text": text}


class DiscoveryPipeline:
    """
    The main discovery pipeline orchestrator.
    Runs all 10 steps of the AI Science Discovery Team.
    """

    def __init__(self, log_callback=None, progress_callback=None):
        self.llm = LLMClient(log_callback=log_callback)
        self.log = log_callback or print
        self.progress = progress_callback or (lambda step, total, msg: None)
        self.results = {}
        self.is_running = False
        self.should_stop = False

        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, "discoveries"), exist_ok=True)

    def _call_agent(self, agent_name, system_prompt, user_message):
        """Call an agent with its configured parameters."""
        config = AGENT_CONFIGS[agent_name]
        
        # Check stop before calling
        if self.should_stop:
            return None
            
        response = self.llm.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            model_id=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
            stop_callback=lambda: self.should_stop
        )
        return response

    def stop(self):
        """Signal the pipeline to stop after the current step."""
        self.should_stop = True
        self.log("⏹️  Stop signal received. Will stop after current step.")

    def _save_progress(self, run_id, step_name=""):
        """Save current progress to disk after each step (incremental save)."""
        progress_path = os.path.join(RESULTS_DIR, f"progress_{run_id}.json")
        latest_path = os.path.join(RESULTS_DIR, "latest_progress.json")
        try:
            data = {
                **self.results,
                "_last_completed_step": step_name,
                "_saved_at": datetime.now().isoformat(),
            }
            for path in [progress_path, latest_path]:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self.log(f"💾 Progress saved after {step_name}")
        except Exception as e:
            self.log(f"⚠️ Could not save progress: {e}")

    def run(self, frontier_problem):
        """
        Run the complete 10-step discovery pipeline.
        
        Args:
            frontier_problem: Description of the frontier science problem to solve.
            
        Returns:
            dict with all results from every step.
        """
        self.is_running = True
        self.should_stop = False
        self.results = {
            "frontier_problem": frontier_problem,
            "timestamp": datetime.now().isoformat(),
            "steps": {},
        }

        total_steps = 10
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1-2: ORCHESTRATOR
            # ═══════════════════════════════════════════════════════════
            self.progress(1, total_steps, "🎯 Step 1-2: Orchestrator — Selecting target and framing task...")
            self.log("\n" + "═" * 70)
            self.log("STEP 1-2: ORCHESTRATOR")
            self.log("Selecting target and framing isolated physics task...")
            self.log("═" * 70)

            orchestrator_response = self._call_agent(
                "orchestrator",
                ORCHESTRATOR_SYSTEM,
                ORCHESTRATOR_USER_TEMPLATE.format(problem_description=frontier_problem),
            )

            if not orchestrator_response or self.should_stop:
                self.log("❌ Orchestrator failed or stopped")
                return self._save_results(run_id)

            orchestrator_data = safe_json_parse(orchestrator_response)
            self.results["steps"]["orchestrator"] = {
                "raw_response": orchestrator_response,
                "parsed": orchestrator_data,
            }
            self.log(f"📋 Selected target: {orchestrator_data.get('selected_target', 'Unknown')}")
            self.log(f"📋 Task: {orchestrator_data.get('task_description', orchestrator_response[:200])}")
            self._save_progress(run_id, "Step 1-2: Orchestrator")

            # Extract fields for next step
            task_desc = orchestrator_data.get("task_description", orchestrator_response)
            target_props = orchestrator_data.get("target_properties", "Not specified")
            known_constraints = orchestrator_data.get("known_constraints", "None specified")

            if self.should_stop:
                return self._save_results(run_id)

            # ═══════════════════════════════════════════════════════════
            # STEP 3: HYPOTHESIS GENERATOR
            # ═══════════════════════════════════════════════════════════
            self.progress(2, total_steps, f"💡 Step 3: Generating {NUM_HYPOTHESES} hypotheses...")
            self.log("\n" + "═" * 70)
            self.log(f"STEP 3: HYPOTHESIS GENERATOR — Generating {NUM_HYPOTHESES} approaches")
            self.log("═" * 70)

            hypothesis_system = HYPOTHESIS_GENERATOR_SYSTEM.format(num_hypotheses=NUM_HYPOTHESES)
            hypothesis_user = HYPOTHESIS_GENERATOR_USER_TEMPLATE.format(
                task_description=task_desc,
                target_properties=target_props,
                known_constraints=known_constraints,
                num_hypotheses=NUM_HYPOTHESES,
            )

            hypothesis_response = self._call_agent(
                "hypothesis_generator", hypothesis_system, hypothesis_user,
            )

            if not hypothesis_response or self.should_stop:
                self.log("❌ Hypothesis generator failed or stopped")
                return self._save_results(run_id)

            hypotheses_data = safe_json_parse(hypothesis_response)
            # Normalize to list
            if isinstance(hypotheses_data, dict):
                hypotheses_list = hypotheses_data.get("approaches", hypotheses_data.get("hypotheses", [hypotheses_data]))
            elif isinstance(hypotheses_data, list):
                hypotheses_list = hypotheses_data
            else:
                hypotheses_list = [{"raw": hypothesis_response}]

            self.results["steps"]["hypotheses"] = {
                "raw_response": hypothesis_response,
                "parsed": hypotheses_list,
            }
            self.log(f"💡 Generated {len(hypotheses_list)} hypotheses")
            for i, h in enumerate(hypotheses_list):
                name = h.get("name", h.get("approach_id", f"Approach {i + 1}"))
                self.log(f"   {i + 1}. {name}")
            self._save_progress(run_id, "Step 3: Hypotheses")

            if self.should_stop:
                return self._save_results(run_id)

            # ═══════════════════════════════════════════════════════════
            # STEPS 4-6: FOR EACH HYPOTHESIS
            # ═══════════════════════════════════════════════════════════
            all_approach_results = []

            for h_idx, hypothesis in enumerate(hypotheses_list):
                if self.should_stop:
                    break

                approach_name = hypothesis.get("name", f"Approach {h_idx + 1}")
                core_mechanism = hypothesis.get("core_mechanism", "Not specified")
                description = hypothesis.get("description", str(hypothesis))
                conditions = json.dumps(hypothesis.get("conditions", {}), indent=2)
                physics_basis = hypothesis.get("physics_basis", "Not specified")

                self.log(f"\n{'─' * 50}")
                self.log(f"PROCESSING APPROACH {h_idx + 1}/{len(hypotheses_list)}: {approach_name}")
                self.log(f"{'─' * 50}")

                # ─────────────────────────────────────────────────────
                # STEP 4: STEP DECOMPOSER
                # ─────────────────────────────────────────────────────
                self.progress(
                    3 + (h_idx * 3 / len(hypotheses_list)),
                    total_steps,
                    f"🔬 Step 4: Decomposing approach {h_idx + 1}: {approach_name}..."
                )
                self.log(f"\n  STEP 4: Decomposing into atomic physics steps...")

                decomposer_user = STEP_DECOMPOSER_USER_TEMPLATE.format(
                    approach_name=approach_name,
                    core_mechanism=core_mechanism,
                    description=description,
                    conditions=conditions,
                    physics_basis=physics_basis,
                )

                decomposer_response = self._call_agent(
                    "step_decomposer", STEP_DECOMPOSER_SYSTEM, decomposer_user,
                )

                if not decomposer_response or self.should_stop:
                    self.log(f"  ⚠️ Step decomposer failed for approach {h_idx + 1}")
                    all_approach_results.append({
                        "approach": hypothesis,
                        "status": "decomposer_failed",
                    })
                    continue

                steps_data = safe_json_parse(decomposer_response)
                if isinstance(steps_data, dict):
                    steps_list = steps_data.get("steps", [steps_data])
                elif isinstance(steps_data, list):
                    steps_list = steps_data
                else:
                    steps_list = [{"raw": decomposer_response}]

                self.log(f"  📐 Decomposed into {len(steps_list)} atomic steps")

                # ─────────────────────────────────────────────────────
                # STEP 5: PHYSICS ORACLE (for each decontextualized question)
                # ─────────────────────────────────────────────────────
                self.progress(
                    4 + (h_idx * 3 / len(hypotheses_list)),
                    total_steps,
                    f"⚛️  Step 5: Physics Oracle validating steps for approach {h_idx + 1}..."
                )
                self.log(f"\n  STEP 5: Physics Oracle — Validating each step from first principles...")

                validated_steps = []
                for s_idx, step in enumerate(steps_list):
                    if self.should_stop:
                        break

                    question = step.get("standalone_question",
                                       step.get("physics_question",
                                                step.get("raw", str(step))))
                    self.log(f"    ⚛️  Question {s_idx + 1}/{len(steps_list)}: {question[:100]}...")

                    oracle_user = PHYSICS_ORACLE_USER_TEMPLATE.format(question=question)
                    oracle_response = self._call_agent(
                        "physics_oracle", PHYSICS_ORACLE_SYSTEM, oracle_user,
                    )

                    oracle_data = safe_json_parse(oracle_response) if oracle_response else None
                    is_possible = True  # Default to possible
                    if oracle_data and isinstance(oracle_data, dict):
                        is_possible = oracle_data.get("physically_possible", True)

                    validated_steps.append({
                        "step": step,
                        "question": question,
                        "oracle_response": oracle_response,
                        "oracle_parsed": oracle_data,
                        "physically_possible": is_possible,
                    })

                    status = "✅ POSSIBLE" if is_possible else "❌ VIOLATES PHYSICS"
                    self.log(f"      {status}")

                # ─────────────────────────────────────────────────────
                # STEP 6: CHAIN ASSEMBLER
                # ─────────────────────────────────────────────────────
                self.progress(
                    5 + (h_idx * 3 / len(hypotheses_list)),
                    total_steps,
                    f"🔗 Step 6: Assembling chain for approach {h_idx + 1}..."
                )
                self.log(f"\n  STEP 6: Chain Assembler — Building coherent pathway...")

                steps_with_validations = ""
                for v_step in validated_steps:
                    steps_with_validations += f"\n--- Step ---\n"
                    steps_with_validations += f"Original: {v_step['step']}\n"
                    steps_with_validations += f"Physics Question: {v_step['question']}\n"
                    steps_with_validations += f"Physics Answer: {v_step.get('oracle_response', 'No response')[:500]}\n"
                    steps_with_validations += f"Physically Possible: {v_step['physically_possible']}\n"

                assembler_user = CHAIN_ASSEMBLER_USER_TEMPLATE.format(
                    approach_name=approach_name,
                    steps_with_validations=steps_with_validations,
                )

                assembler_response = self._call_agent(
                    "chain_assembler", CHAIN_ASSEMBLER_SYSTEM, assembler_user,
                )

                assembler_data = safe_json_parse(assembler_response) if assembler_response else None
                chain_status = "UNKNOWN"
                if assembler_data and isinstance(assembler_data, dict):
                    chain_status = assembler_data.get("chain_status", "UNKNOWN")

                self.log(f"  🔗 Chain status: {chain_status}")

                all_approach_results.append({
                    "approach": hypothesis,
                    "decomposed_steps": steps_list,
                    "validated_steps": validated_steps,
                    "chain_assembly": {
                        "raw_response": assembler_response,
                        "parsed": assembler_data,
                        "status": chain_status,
                    },
                    "status": "chain_assembled",
                })

            self.results["steps"]["approach_results"] = all_approach_results
            self._save_progress(run_id, "Steps 4-6: Approach Validation")

            if self.should_stop:
                return self._save_results(run_id)

            # ═══════════════════════════════════════════════════════════
            # STEP 7: ENGINEERING PROPOSER (for valid/fixable chains)
            # ═══════════════════════════════════════════════════════════
            self.progress(6, total_steps, "🏗️  Step 7: Engineering Proposer...")
            self.log("\n" + "═" * 70)
            self.log("STEP 7: ENGINEERING PROPOSER")
            self.log("═" * 70)

            engineering_results = []
            valid_approaches = [
                r for r in all_approach_results
                if r.get("status") == "chain_assembled"
                and r.get("chain_assembly", {}).get("status") in ("VALID", "FIXABLE", "UNKNOWN")
            ]

            for v_idx, approach_result in enumerate(valid_approaches):
                if self.should_stop:
                    break

                approach_name = approach_result["approach"].get("name", f"Approach {v_idx + 1}")
                self.log(f"\n  Engineering proposals for: {approach_name}")

                assembly = approach_result.get("chain_assembly", {}).get("parsed", {})
                assembled_pathway = assembly.get("assembled_pathway", str(assembly)) if isinstance(assembly, dict) else str(assembly)
                overall_conditions = json.dumps(
                    assembly.get("overall_conditions", {}) if isinstance(assembly, dict) else {},
                    indent=2
                )

                eng_user = ENGINEERING_PROPOSER_USER_TEMPLATE.format(
                    assembled_pathway=assembled_pathway,
                    overall_conditions=overall_conditions,
                )

                eng_response = self._call_agent(
                    "engineering_proposer", ENGINEERING_PROPOSER_SYSTEM, eng_user,
                )

                eng_data = safe_json_parse(eng_response) if eng_response else None
                engineering_results.append({
                    "approach_name": approach_name,
                    "approach_result": approach_result,
                    "engineering": {
                        "raw_response": eng_response,
                        "parsed": eng_data,
                    },
                })
                self.log(f"  🏗️  Engineering proposals generated for {approach_name}")

            self.results["steps"]["engineering"] = engineering_results
            self._save_progress(run_id, "Step 7: Engineering")

            if self.should_stop:
                return self._save_results(run_id)

            # ═══════════════════════════════════════════════════════════
            # STEP 8: REQUIREMENT CHALLENGER (iterative loop)
            # ═══════════════════════════════════════════════════════════
            self.progress(7, total_steps, f"🔄 Step 8: Requirement Challenger ({CHALLENGE_ITERATIONS} iterations)...")
            self.log("\n" + "═" * 70)
            self.log(f"STEP 8: REQUIREMENT CHALLENGER — {CHALLENGE_ITERATIONS} iterations")
            self.log("═" * 70)

            challenge_results = []
            for eng_idx, eng_result in enumerate(engineering_results):
                if self.should_stop:
                    break

                approach_name = eng_result["approach_name"]
                self.log(f"\n  Challenging requirements for: {approach_name}")

                engineering_proposal = eng_result["engineering"].get("raw_response", "No proposal")
                assembly = eng_result["approach_result"].get("chain_assembly", {}).get("parsed", {})
                assembled_pathway = assembly.get("assembled_pathway", str(assembly)) if isinstance(assembly, dict) else str(assembly)

                iteration_results = []
                previous_challenges = "None yet — this is the first iteration."

                for iteration in range(CHALLENGE_ITERATIONS):
                    if self.should_stop:
                        break

                    self.log(f"    🔄 Challenge iteration {iteration + 1}/{CHALLENGE_ITERATIONS}")

                    challenger_system = REQUIREMENT_CHALLENGER_SYSTEM.format(
                        iteration=iteration + 1,
                        max_iterations=CHALLENGE_ITERATIONS,
                    )
                    challenger_user = REQUIREMENT_CHALLENGER_USER_TEMPLATE.format(
                        engineering_proposal=engineering_proposal,
                        assembled_pathway=assembled_pathway,
                        previous_challenges=previous_challenges,
                    )

                    challenge_response = self._call_agent(
                        "requirement_challenger", challenger_system, challenger_user,
                    )

                    challenge_data = safe_json_parse(challenge_response) if challenge_response else None

                    # Validate each challenge with the Physics Oracle
                    validated_challenges = []
                    if challenge_data and isinstance(challenge_data, dict):
                        challenges = challenge_data.get("challenges", [])
                        for c in challenges:
                            if self.should_stop:
                                break
                            physics_q = c.get("physics_question_to_validate", "")
                            if physics_q:
                                self.log(f"      ⚛️  Validating: {physics_q[:80]}...")
                                oracle_resp = self._call_agent(
                                    "physics_oracle",
                                    PHYSICS_ORACLE_SYSTEM,
                                    PHYSICS_ORACLE_USER_TEMPLATE.format(question=physics_q),
                                )
                                oracle_parsed = safe_json_parse(oracle_resp) if oracle_resp else None
                                validated_challenges.append({
                                    "challenge": c,
                                    "physics_validation": oracle_resp,
                                    "physics_parsed": oracle_parsed,
                                })
                            else:
                                validated_challenges.append({
                                    "challenge": c,
                                    "physics_validation": "No physics question provided",
                                })

                    iteration_result = {
                        "iteration": iteration + 1,
                        "challenges": challenge_response,
                        "challenges_parsed": challenge_data,
                        "validated_challenges": validated_challenges,
                    }
                    iteration_results.append(iteration_result)

                    # Build context for next iteration
                    previous_challenges += f"\n\nIteration {iteration + 1} challenges:\n{challenge_response}\n"
                    prev_validations = ""
                    for vc in validated_challenges:
                        ch = vc.get("challenge", {})
                        prev_validations += f"\nChallenge: {ch.get('challenge_question', '?')}\n"
                        prev_validations += f"Physics validation: {str(vc.get('physics_validation', ''))[:300]}\n"
                    previous_challenges += f"Physics validations:\n{prev_validations}"

                challenge_results.append({
                    "approach_name": approach_name,
                    "iterations": iteration_results,
                })

            self.results["steps"]["challenges"] = challenge_results
            self._save_progress(run_id, "Step 8: Challenges")

            if self.should_stop:
                return self._save_results(run_id)

            # ═══════════════════════════════════════════════════════════
            # STEP 9: OVERSEER / SYNTHESIZER
            # ═══════════════════════════════════════════════════════════
            self.progress(8, total_steps, "🔍 Step 9: Overseer synthesizing...")
            self.log("\n" + "═" * 70)
            self.log("STEP 9: OVERSEER / SYNTHESIZER")
            self.log("═" * 70)

            # Build comprehensive summary of all approaches
            all_approaches_summary = ""
            for idx, ar in enumerate(all_approach_results):
                approach = ar.get("approach", {})
                chain = ar.get("chain_assembly", {})
                all_approaches_summary += f"\n{'─' * 40}\n"
                all_approaches_summary += f"APPROACH {idx + 1}: {approach.get('name', 'Unknown')}\n"
                all_approaches_summary += f"Mechanism: {approach.get('core_mechanism', 'Unknown')}\n"
                all_approaches_summary += f"Description: {approach.get('description', 'Unknown')}\n"
                all_approaches_summary += f"Chain Status: {chain.get('status', 'Unknown')}\n"

                # Include validated steps summary
                for vs in ar.get("validated_steps", []):
                    q = vs.get("question", "?")
                    possible = vs.get("physically_possible", "?")
                    all_approaches_summary += f"  Step: {q[:100]} → {'✅' if possible else '❌'}\n"

                # Include chain assembly details
                chain_parsed = chain.get("parsed", {})
                if isinstance(chain_parsed, dict):
                    pathway = chain_parsed.get("assembled_pathway", "")
                    if pathway:
                        all_approaches_summary += f"Assembled Pathway: {pathway}\n"

            # Build challenge summary
            all_challenges_summary = ""
            for cr in challenge_results:
                all_challenges_summary += f"\n{'─' * 40}\n"
                all_challenges_summary += f"Challenges for: {cr['approach_name']}\n"
                for iteration in cr.get("iterations", []):
                    all_challenges_summary += f"\n  Iteration {iteration['iteration']}:\n"
                    for vc in iteration.get("validated_challenges", []):
                        ch = vc.get("challenge", {})
                        all_challenges_summary += f"    Q: {ch.get('challenge_question', '?')}\n"
                        all_challenges_summary += f"    Physics: {str(vc.get('physics_validation', ''))[:200]}\n"

            overseer_user = OVERSEER_USER_TEMPLATE.format(
                original_task=task_desc,
                all_approaches_summary=all_approaches_summary,
                all_challenges_summary=all_challenges_summary,
            )

            overseer_response = self._call_agent(
                "overseer", OVERSEER_SYSTEM, overseer_user,
            )

            overseer_data = safe_json_parse(overseer_response) if overseer_response else None
            self.results["steps"]["overseer"] = {
                "raw_response": overseer_response,
                "parsed": overseer_data,
            }
            self.log("🔍 Overseer synthesis complete")
            self._save_progress(run_id, "Step 9: Overseer")

            if self.should_stop:
                return self._save_results(run_id)

            # ═══════════════════════════════════════════════════════════
            # STEP 10: FINAL EVALUATOR
            # ═══════════════════════════════════════════════════════════
            self.progress(9, total_steps, "📝 Step 10: Final Evaluator — Writing discovery thesis...")
            self.log("\n" + "═" * 70)
            self.log("STEP 10: FINAL EVALUATOR — Writing discovery thesis")
            self.log("═" * 70)

            # Extract top proposals from overseer
            top_proposals_text = ""
            if overseer_data and isinstance(overseer_data, dict):
                synthesis = overseer_data.get("synthesis", [])
                for s in synthesis[:NUM_FINAL_PROPOSALS]:
                    top_proposals_text += f"\n{'─' * 40}\n"
                    top_proposals_text += f"Rank: {s.get('rank', '?')}\n"
                    top_proposals_text += f"Name: {s.get('name', 'Unknown')}\n"
                    top_proposals_text += f"Pathway: {s.get('complete_pathway', 'Unknown')}\n"
                    top_proposals_text += f"Physics Confidence: {s.get('physics_confidence', '?')}\n"
                    top_proposals_text += f"Engineering: {s.get('engineering_feasibility', '?')}\n"
                    top_proposals_text += f"Innovation: {s.get('key_innovation', '?')}\n"
            else:
                top_proposals_text = overseer_response or "No synthesis available"

            evaluator_user = FINAL_EVALUATOR_USER_TEMPLATE.format(
                top_proposals=top_proposals_text,
                original_problem=frontier_problem,
            )

            evaluator_response = self._call_agent(
                "final_evaluator", FINAL_EVALUATOR_SYSTEM, evaluator_user,
            )

            self.results["steps"]["final_thesis"] = {
                "raw_response": evaluator_response,
            }
            self.log("📝 Discovery thesis written!")

            # ═══════════════════════════════════════════════════════════
            # SAVE RESULTS
            # ═══════════════════════════════════════════════════════════
            self.progress(10, total_steps, "💾 Saving results...")
            return self._save_results(run_id)

        except Exception as e:
            self.log(f"❌ Pipeline error: {e}")
            import traceback
            self.log(traceback.format_exc())
            return self._save_results(run_id)
        finally:
            self.is_running = False

    def _save_results(self, run_id):
        """Save all results to files."""
        # Save full JSON results
        json_path = os.path.join(RESULTS_DIR, f"discovery_{run_id}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            self.log(f"💾 Full results saved to: {json_path}")
        except Exception as e:
            self.log(f"⚠️ Error saving JSON: {e}")

        # Save human-readable thesis
        thesis = self.results.get("steps", {}).get("final_thesis", {}).get("raw_response", "")
        if thesis:
            thesis_path = os.path.join(RESULTS_DIR, "discoveries", f"thesis_{run_id}.md")
            try:
                with open(thesis_path, "w", encoding="utf-8") as f:
                    f.write(f"# Discovery Thesis\n")
                    f.write(f"**Generated**: {self.results.get('timestamp', 'Unknown')}\n\n")
                    f.write(f"## Frontier Problem\n{self.results.get('frontier_problem', 'Unknown')}\n\n")
                    f.write(f"## Discovery\n{thesis}\n")
                self.log(f"📄 Thesis saved to: {thesis_path}")
            except Exception as e:
                self.log(f"⚠️ Error saving thesis: {e}")

        # Save pipeline summary
        summary = self._generate_summary()
        summary_path = os.path.join(RESULTS_DIR, f"summary_{run_id}.md")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            self.log(f"📋 Summary saved to: {summary_path}")
        except Exception as e:
            self.log(f"⚠️ Error saving summary: {e}")

        return self.results

    def _generate_summary(self):
        """Generate a human-readable summary of the pipeline run."""
        lines = []
        lines.append("# 🔬 AI Science Discovery — Pipeline Summary\n")
        lines.append(f"**Timestamp**: {self.results.get('timestamp', 'Unknown')}\n")
        lines.append(f"## Frontier Problem\n{self.results.get('frontier_problem', 'Unknown')}\n")

        steps = self.results.get("steps", {})

        # Orchestrator
        orch = steps.get("orchestrator", {}).get("parsed", {})
        if isinstance(orch, dict):
            lines.append(f"## Selected Target\n{orch.get('selected_target', 'Unknown')}\n")
            lines.append(f"**Task**: {orch.get('task_description', 'Unknown')}\n")

        # Hypotheses
        hypotheses = steps.get("hypotheses", {}).get("parsed", [])
        if hypotheses:
            lines.append(f"## Hypotheses Generated: {len(hypotheses)}\n")
            for i, h in enumerate(hypotheses):
                name = h.get("name", f"Approach {i + 1}")
                lines.append(f"### {i + 1}. {name}")
                lines.append(f"- **Mechanism**: {h.get('core_mechanism', 'Unknown')}")
                lines.append(f"- **Physics Basis**: {h.get('physics_basis', 'Unknown')}\n")

        # Approach results
        approach_results = steps.get("approach_results", [])
        if approach_results:
            lines.append("## Approach Validation Results\n")
            for ar in approach_results:
                approach = ar.get("approach", {})
                chain = ar.get("chain_assembly", {})
                name = approach.get("name", "Unknown")
                status = chain.get("status", ar.get("status", "Unknown"))
                lines.append(f"### {name}")
                lines.append(f"- **Chain Status**: {status}")

                valid_count = sum(1 for vs in ar.get("validated_steps", []) if vs.get("physically_possible"))
                total_count = len(ar.get("validated_steps", []))
                lines.append(f"- **Physics Steps Validated**: {valid_count}/{total_count}\n")

        # Overseer synthesis
        overseer = steps.get("overseer", {}).get("parsed", {})
        if isinstance(overseer, dict):
            synthesis = overseer.get("synthesis", [])
            if synthesis:
                lines.append("## Top Synthesized Solutions\n")
                for s in synthesis:
                    lines.append(f"### Rank {s.get('rank', '?')}: {s.get('name', 'Unknown')}")
                    lines.append(f"- **Physics Confidence**: {s.get('physics_confidence', '?')}")
                    lines.append(f"- **Key Innovation**: {s.get('key_innovation', '?')}")
                    lines.append(f"- **Pathway**: {s.get('complete_pathway', '?')}\n")

        # Final thesis reference
        if steps.get("final_thesis", {}).get("raw_response"):
            lines.append("## Discovery Thesis\nSee full thesis in the discoveries/ folder.\n")

        return "\n".join(lines)
