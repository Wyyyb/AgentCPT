import json
import random
import os
from glm_api_0804 import get_glm45_response


prompt_template = """## Objective:
As an expert AI analyst specializing in detecting implicit agentic structures in non-agentic text data (such as narratives, reports, or descriptive passages), analyze the given_text to assign quantitative scores across 12 core dimensions. These dimensions identify subtle elements that could enhance the potential of language models for agentic capabilities during continual pretraining (CPT), such as implicit sequential reasoning, adaptive logic, and dynamic interactions. Focus on how these implicit structures—embedded in non-instructional, pretraining-style data—can foster agent-like skills (e.g., planning, flexibility) without explicit agentic formatting. Adhere strictly to the provided definitions and schemata. Apply a strict scoring schema: aim for approximately 90% of evaluated texts to score 3 or below on most dimensions, reserving 4 for texts with very high implicit depth and 5 for exceptional, benchmark-quality examples that profoundly exemplify agentic potential (representing <1% of texts).

## Input:
A `given_text` for evaluation.

### Task: Dimensional Scoring
Assign an integer score (1-5) for each dimension based strictly on its schema. For each dimension, provide:
1. A detailed explanation, including direct quotes from the text as evidence, justifying the score and highlighting how the implicit structures contribute to agentic potential.
2. A list of 3-5 key phrases that support the score.
Output exclusively as a single, valid JSON object (no additional text, comments, or wrappers). Use the exact dimension names as keys (e.g., "sequence_decision_density"). The structure must be:

{
  "sequence_decision_density": {"score": <1-5>, "explanation": "<detailed reason with quotes from text>", "key_phrases": ["<phrase1>", "<phrase2>", ...]},  
  "conditional_branching_complexity": {...},  
  "goal_orientedness": {...},  
  "uncertainty_and_exploration": {...},  
  "interactivity": {...},  
  "feedback_loops": {...},  
  "multi_agent_collaboration": {...},  
  "resource_management": {...},  
  "risk_assessment_and_tradeoffs": {...},  
  "long_term_planning_horizon": {...},  
  "tool_invocation_simulation": {...},  
  "adaptation_to_changes": {...}  
}

**General Scoring Principle:** Assign the score that best describes the text's implicit agentic depth. If the text falls between scores, default to the lower one unless it unequivocally meets most criteria for the higher. Scores of 4 and 5 are rare and reserved for texts with profound, implicit agentic sophistication that could serve as high-value CPT data for enhancing agent potential. Focus exclusively on implicit structures; purely descriptive or static text (e.g., factual lists without implied dynamics) should rarely exceed 2. Ignore explicit agentic elements if present, as the goal is to detect hidden potential in non-agentic data.

#### Dimensions & Scoring Schemata:

**1. Sequence Decision Density** (sequence_decision_density)  
Assess the density and depth of implicit chained decision points or sequential actions, such as in a narrative where events imply layered planning (e.g., "The explorer assessed the terrain, then chose a path based on visibility, adjusting for potential hazards along the route"). This implicitly trains multi-step reasoning for agents by embedding coherent action chains in non-agentic text.  
* **1:** Minimal or no implicit sequences; isolated events require significant inference to suggest any chaining, offering only rudimentary hints for basic agentic flow.  
* **2:** Sparse implicit chains with basic linking (e.g., 1-2 loosely connected steps), somewhat imprecise but adequate for simple sequential implications in contexts like routine descriptions.  
* **3:** Moderate density with coherent, implicit chaining (e.g., 3-4 intertwined steps forming a logical progression), effectively implying agentic planning depth.  
* **4:** High density with complex, precise implicit sequences (e.g., multiple intertwined chains demonstrating strategic layering), ideally suited for training sophisticated multi-step agent structures.  
* **5:** Exceptional mastery with profound, densely interwoven implicit chains (e.g., elegant, multi-layered progressions that benchmark implicit agentic elegance), representing rare texts with unparalleled potential for CPT.

**2. Conditional Branching Complexity** (conditional_branching_complexity)  
Assess the presence and nesting depth of implicit conditional logic or branching paths, such as in analytical text where outcomes depend on variables (e.g., "Should market volatility increase, investors might pivot to bonds; otherwise, equities remain viable, with further diversification if inflation rises"). This fosters implicit adaptive reasoning in agents.  
* **1:** Negligible implicit conditions; flat narratives hinder branching inference, providing only basic adaptability hints with effort.  
* **2:** Simple single-level implicit branches (e.g., basic if-then hints), occasionally imprecise but okay for mild conditional elements in straightforward scenarios.  
* **3:** Moderate nesting (e.g., 2-3 levels of implicit logic), clearly presenting branching paths that support agentic decision trees.  
* **4:** Deep nesting with precise, multi-level alternatives (e.g., 4+ layers of conditional depth), perfectly defining adaptive structures for agent training.  
* **5:** Masterful complexity with nuanced, highly nested implicit branches (e.g., flawless integration of multiple contingencies), exemplifying exceptional agentic reasoning potential in <1% of texts.

**3. Goal-Orientedness** (goal_orientedness)  
Assess how clearly the text implies a goal with directed, implicit steps, such as in a historical account of a project's progression (e.g., "The team gathered data to inform the design, iterated prototypes based on tests, and scaled production upon validation"). This builds implicit pursuit mechanisms in agents.  
* **1:** Vague or absent implicit goals; isolated elements require effort to infer orientation, offering minimal pursuit hints.  
* **2:** Vague implicit goals with 1-2 basic steps, somewhat clunky but adequate for simple task implications.  
* **3:** Clear implicit goals with 3-4 detailed, directed steps, effectively communicating agentic pursuit.  
* **4:** Strong, multi-faceted implicit goals with integrated steps (e.g., 5+ cohesive elements), precisely structured for goal-directed agent planning.  
* **5:** Outstanding mastery with deeply nuanced, integrated implicit steps (e.g., elegant long-term orientation), benchmarking profound agentic potential in rare texts.

**4. Uncertainty and Exploration** (uncertainty_and_exploration)  
Assess levels of implicit unpredictability and alternative paths, such as in exploratory narratives (e.g., "The outcome hinged on variables: one route promised quick gains but hidden risks, another offered steady progress through trial and error"). This encourages implicit curiosity and probing in agents.  
* **1:** Fully deterministic with no implicit uncertainty; static elements make exploration hard to infer, providing only basic hints.  
* **2:** Mild implicit uncertainty with 1-2 options, somewhat inconsistent but okay for simple multi-path elements.  
* **3:** Moderate uncertainty (e.g., 3-4 implicit paths with exploration cues), effectively supporting agentic adaptability.  
* **4:** High uncertainty with strong, precise implicit alternatives (e.g., 5+ paths emphasizing trial), ideal for training exploratory agents.  
* **5:** Exceptional mastery with nuanced, profound implicit alternatives (e.g., flawless emphasis on deep uncertainty), representing <1% of texts with benchmark exploration potential.

**5. Interactivity** (interactivity)  
Assess implicit simulated interactions between entities, such as in dialogues or group dynamics within narratives (e.g., "The advisor proposed a strategy, the leader countered with concerns, leading to a refined consensus through back-and-forth refinement"). This develops implicit collaboration in agents.  
* **1:** No implicit interactions; static descriptions require effort to infer dynamics, offering basic reciprocity hints.  
* **2:** Simple one-way implicit exchanges, occasionally awkward but adequate for basic collaborative implications.  
* **3:** Moderate multi-turn implicit reciprocity, effectively simulating entity dynamics for agentic contexts.  
* **4:** Dense, sophisticated implicit multi-entity interactions with reciprocal depth, perfectly defining collaborative structures.  
* **5:** Outstanding mastery with profound, nuanced implicit dynamics (e.g., flawless multi-turn impact), exemplifying rare agentic collaboration potential.

**6. Feedback Loops** (feedback_loops)  
Assess implicit action-result-adjustment cycles, such as in process-oriented text (e.g., "Initial approach yielded partial results, prompting refinements based on discrepancies, ultimately leading to optimized outcomes through iterations"). This builds implicit resilience in agents.  
* **1:** No implicit cycles; static elements hinder loop inference, providing only basic adjustment hints.  
* **2:** Single basic implicit loop, somewhat imprecise but okay for simple iterative elements.  
* **3:** 2-3 iterative implicit loops, effectively presenting action-result-adjustment for agentic learning.  
* **4:** Multiple deep implicit loops (e.g., 3+ with layered adjustments), precisely structured for resilient training.  
* **5:** Exceptional mastery with complex, nuanced implicit cycles (e.g., >3 flawless in depth), representing <1% of texts with profound resilience potential.

**7. Multi-Agent Collaboration** (multi_agent_collaboration)  
Assess implicit cooperative dynamics among entities, such as in team-based accounts (e.g., "Roles were distributed: one analyzed data, another implemented changes, synchronizing efforts to adapt to emerging challenges"). This fosters implicit teamwork in agents.  
* **1:** No implicit collaboration; solo elements offer only basic coordination hints.  
* **2:** Basic implicit pairing, somewhat inconsistent but adequate for simple synergy.  
* **3:** Moderate implicit coordination with defined roles, effectively implying multi-agent dynamics.  
* **4:** Complex multi-party implicit synergy with adaptive adjustments, precisely defined for team-oriented agent training.  
* **5:** Outstanding mastery with profound, nuanced implicit coordination (e.g., flawless execution), exemplifying rare benchmark texts.

**8. Resource Management** (resource_management)  
Assess implicit handling of constraints, such as in optimization scenarios (e.g., "Assets were allocated strategically: prioritizing critical areas while reserving buffers for unforeseen demands, reallocating as needs evolved"). This develops implicit optimization in agents.  
* **1:** No implicit resource mentions; absence hinders allocation inference, providing basic hints.  
* **2:** Basic implicit handling, somewhat clunky but okay for simple trade-offs.  
* **3:** Moderate implicit trade-offs, effectively presenting dynamic resource elements.  
* **4:** Detailed, dynamic implicit optimization with precise structures, ideal for efficient agent training.  
* **5:** Exceptional mastery with profound, nuanced implicit management (e.g., flawless impact), representing <1% of texts.

**9. Risk Assessment and Trade-offs** (risk_assessment_and_tradeoffs)  
Assess implicit weighing of risks and benefits, such as in strategic analyses (e.g., "The venture balanced high potential returns against volatility risks, hedging with diversified options to mitigate downsides"). This sharpens implicit prudence in agents.  
* **1:** No implicit risks; absence makes weighing hard to infer, offering basic hints.  
* **2:** Simple implicit pros/cons, occasionally imprecise but adequate for basic judgment.  
* **3:** Multi-factor implicit weighing, effectively implying balanced agentic caution.  
* **4:** Deep, quantitative implicit trade-offs with precision, perfectly suited for prudent training.  
* **5:** Outstanding mastery with profound, nuanced implicit assessments (e.g., flawless depth), exemplifying rare texts.

**10. Long-Term Planning Horizon** (long_term_planning_horizon)  
Assess implicit extended multi-phase planning, such as in forward-looking narratives (e.g., "Initial research laid the foundation, followed by implementation phases, with long-term scaling contingent on milestones over extended periods"). This cultivates implicit vision in agents.  
* **1:** Immediate-only focus; no implicit phases, providing basic foresight hints.  
* **2:** Short-term implicit phase (e.g., 1 vague extension), somewhat adequate for simple planning.  
* **3:** Medium-term with 2-3 implicit phases, effectively extending agentic horizons.  
* **4:** Long-term spanning 4+ implicit phases with contingencies, precisely structured for visionary training.  
* **5:** Exceptional mastery with >4 nuanced, cross-temporal implicit phases (e.g., flawless foresight), representing <1% of texts.

**11. Tool Invocation Simulation** (tool_invocation_simulation)  
Assess implicit simulation of tool usage, such as in procedural descriptions (e.g., "Data was processed via analytical methods: querying databases for inputs, computing aggregates, and integrating results into broader evaluations"). This mimics implicit tool-calling in agents.  
* **1:** No implicit tools; absence hinders simulation, offering basic hints.  
* **2:** Vague implicit use, somewhat inconsistent but okay for simple integrations.  
* **3:** Detailed implicit invocations with parameters, effectively mimicking agentic tools.  
* **4:** Frequent, sequenced implicit invocations, precisely defined for external agent integration.  
* **5:** Outstanding mastery with profound, chained implicit simulations (e.g., flawless nuance), exemplifying rare benchmark texts.

**12. Adaptation to Changes** (adaptation_to_changes)  
Assess implicit dynamic pivots to disruptions, such as in adaptive accounts (e.g., "Original strategy faltered amid shifts; resources were realigned to alternative paths, with further tweaks based on ongoing feedback"). This builds implicit flexibility in agents.  
* **1:** Static with no implicit adjustments; offering basic resilience hints.  
* **2:** Simple single implicit adjustment, somewhat awkward but adequate for basic pivots.  
* **3:** Moderate multi-change implicit responses, effectively handling disruptions for agentic flexibility.  
* **4:** Profound, layered implicit adaptations to complex shifts, precisely structured.  
* **5:** Exceptional mastery with nuanced, resilient implicit adaptations (e.g., flawless depth), representing <1% of texts.
"""


def filter_long_prompts(tokenizer, messages, prompts_data, max_tokens=32000):
    """过滤掉超过指定token数量的prompts"""
    filtered_messages = []
    filtered_indices = []

    for i, message in enumerate(messages):
        # 计算token数量
        tokens = tokenizer.encode(message)
        token_count = len(tokens)

        if token_count <= max_tokens:
            filtered_messages.append(message)
            filtered_indices.append(i)
        else:
            print(f"过滤掉第 {i} 个prompt，token数量: {token_count}")

    # 同时过滤对应的原始数据
    filtered_data = [prompts_data[i] for i in filtered_indices]

    print(f"原始prompts数量: {len(messages)}")
    print(f"过滤后prompts数量: {len(filtered_messages)}")

    return filtered_messages, filtered_data, filtered_indices

def inference_glm_air(prompts, data):
    for i, prompt in enumerate(prompts):
        each = data[i]
        if each["agent_cpt_dict_0804"].get("GLM_4.5_Air_IASS_Score", None) is not None:
            print("skip it")
            continue
        messages = [{"role": "user", "content": prompt}]
        response = get_glm45_response(messages)
        if response is None:
            print("failed")
            continue
        score = extract_score(response)
        data[i]["agent_cpt_dict_0804"]["GLM_4.5_Air_IASS_Content"] = response
        data[i]["agent_cpt_dict_0804"]["GLM_4.5_Air_IASS_Score"] = score
        save_res(data)


def filter_data_0804(data):
    res = []
    for each in data:
        if "reward_dict" in each:
            m2_reward = {}
            score = each["reward_dict"]
            for k, v in score.items():
                if k.startswith("score"):
                    m2_reward[k] = v
            if len(list(m2_reward.keys())) != 16:
                print("different m2 reward:\n", m2_reward)
                print(each["file_path"])
            ori_dict = each["agent_cpt_dict"]
            each["agent_cpt_dict_0804"] = {"file_path": ori_dict["file_path"],
                                           "dataset_name": ori_dict["dataset_name"]}
            res.append(each)
    random.shuffle(res)
    print("len(res):", len(res))
    with open("../local_data/data_0804/ori_sample_data_0804.json", "w") as f:
        f.write(json.dumps(res, indent=4))
    return res


def save_res(res):
    with open("../local_data/data_0804/res_sample_data_with_glm_score_0804.json", "w") as f:
        f.write(json.dumps(res, indent=4))

def load_model_and_data():
    with open("../local_data/test_data_0731/qwen3_sample_100_each_data_with_IASS.json", "r") as f:
        data = json.load(f)

    if os.path.exists("../local_data/data_0804/res_sample_data_with_glm_score_0804.json"):
        with open("../local_data/data_0804/res_sample_data_with_glm_score_0804.json", "r") as f:
            data = json.load(f)
        print("len(data):", len(data))
    else:
        data = filter_data_0804(data)

    prompts = []
    # data = data[:100]
    for each in data:
        text = each["data"][0]["text"]
        prompt = prompt_template.replace("A `given_text` for evaluation.", text)
        prompts.append(prompt)
    return prompts, data


def extract_score(output):
    if output.startswith("```json\n"):
        output = output[len("```json\n"):]
    if output.endswith("```"):
        output = output[:-len("```")]
    if "</think>\n\n" in output:
        content = output.split("</think>\n\n")
        res = content[1]
    else:
        res = output
    try:
        score = json.loads(res)
    except json.decoder.JSONDecodeError:
        print("json.decoder.JSONDecodeError", res)
        score = None
    except Exception as e:
        print("Exception", e)
        score = None
    return score


def main():
    prompts, data = load_model_and_data()
    inference_glm_air(prompts, data)


if __name__ == "__main__":
    main()


