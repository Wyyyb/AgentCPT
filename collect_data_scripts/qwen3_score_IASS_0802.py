import json
from vllm import LLM, SamplingParams


prompt_template = """## Objective:
As an expert AI analyst specializing in detecting implicit agentic structures in non-agent text data (such as books or articles), analyze `given_text` to assign quantitative scores for core agentic dimensions. These dimensions uncover elements that could enhance AI agent potential, such as sequential planning, adaptive decision-making, and collaborative dynamics. Adhere strictly to the provided definitions and schemata. The scoring schema has been revised to be stricter, aiming for approximately 95% of evaluated texts to score 3 or below on most dimensions, reserving scores 4 and 5 for genuinely high-quality and exceptional texts that exemplify profound agentic depth, respectively.

## Input:
A `given_text` for evaluation.

### Task: Dimensional Scoring
Assign an integer score (1-5) for each dimension based strictly on its schema. For each dimension, also provide a detailed explanation (including quotes from the text as evidence) and a list of key phrases that support the score. Output everything in a single JSON object with the following structure:  
`{  
  "dimension1": {"score": <1-5>, "explanation": "<detailed reason with quotes from text>", "key_phrases": ["<phrase1>", "<phrase2>"]},  
  "dimension2": {...},  
  ...  
  "dimension12": {...}  
}`  
Replace "dimension1" etc. with the actual dimension names (e.g., "sequence_decision_density").

**General Scoring Principle:** Assign the score best describing the text. If between scores, choose the lower unless it unequivocally meets most criteria for the higher. Scores 4 and 5 should be used sparingly, reserved for texts that truly exemplify very high quality or outstanding excellence in that dimension. Focus on implicit agentic structures; pure descriptive or non-agentic text (e.g., "The sky is blue") should rarely exceed 2.

#### Dimensions & Scoring Schemata:

**1. Sequence Decision Density**  
Evaluate how densely the text contains chained decision points or sequential actions (e.g., "first check inventory, then move forward, if safe proceed"). This is crucial for implicit training of multi-step decision-making in agents.  
* **1:** The text has minimal or no sequences, with isolated actions or pure descriptions that barely suggest any chaining, requiring significant effort to infer even basic decision density. It is "not quite good but ok" for very rudimentary agentic hints.  
* **2:** The text shows sparse chains (e.g., 1-2 decisions per 100 tokens, like "Wake up, eat, work"), with some awkward or imprecise sequencing. It is "ok" for basic multi-step implications, and some might find it adequately sequential for simple contexts.  
* **3:** The text exhibits "great" moderate density with 3-4 linked decisions, some chaining, clearly forming coherent sequences that effectively imply agentic planning.  
* **4:** Demonstrates "very good" high density with 5-6 intertwined sequences, forming complex chains with precise, impactful linking, perfectly defining effective multi-step agentic structures.  
* **5:** Achieves an "outstanding" and unparalleled level of sequence mastery, with >6 deeply intertwined, complex chains per 100 tokens, showcasing profound density and elegance. So exceptionally crafted it could serve as a benchmark for agentic training; representing the <1% of truly remarkable texts.

**2. Conditional Branching Complexity**  
Evaluate the presence and nesting depth of conditional logic or branching paths (e.g., "If raining, take umbrella; else walk; if windy, also wear coat"). This fosters implicit branching in agent reasoning.  
* **1:** The text has no or negligible conditions, with flat descriptions that hinder any branching inference, but basic logic might be decipherable with effort. It is "not quite good but ok" for very basic adaptability hints.  
* **2:** The text includes simple single-level branches (e.g., "If hungry, eat"), often inconsistent or imprecise. It is "ok" for conveying mild conditional elements, and some might find it adequately branching for simpler scenarios.  
* **3:** The text is clearly "great" with moderate nesting (2-3 levels), employing effective conditional logic that presents branching paths understandably for agentic contexts.  
* **4:** Exhibits "very good" deep nesting with 4 levels and alternatives, using precise and unambiguous conditions for a knowledgeable audience, perfectly defining adaptive decision trees.  
* **5:** Demonstrates an "outstanding" masterful complexity with multi-level, nuanced nesting (>4 levels), flawlessly employed for exceptional agentic reasoning; representing the <1% of authoritative texts.

**3. Goal-Orientedness**  
Evaluate how clearly the text describes a goal with directed steps (e.g., "To bake a cake: mix ingredients, bake at 180°C, cool"). This builds implicit goal pursuit in agents.  
* **1:** The text has no or vague goals, with isolated descriptions requiring effort to infer any orientation. It is "not quite good but ok" for very basic pursuit hints.  
* **2:** The text shows vague goals with 1-2 steps, often imprecise or clunky. It is "ok" for simple goal implications, and some might find it adequately oriented for basic tasks.  
* **3:** The text exhibits "great" clarity with 3-4 detailed steps toward a clear goal, effectively communicating directed pursuit.  
* **4:** Demonstrates "very good" strong, multi-faceted goals with 5 integrated steps, precisely and impactfully structured for perfect goal-directed planning.  
* **5:** Achieves "outstanding" mastery with >5 deeply integrated, nuanced steps, showcasing profound elegance in goal pursuit; representing the <1% of benchmark texts.

**4. Uncertainty and Exploration**  
Evaluate levels of unpredictability and multiple paths (e.g., "Outcome unknown: path A might lead to success, path B to failure, explore both"). This encourages implicit curiosity in agents.  
* **1:** The text is fully deterministic with no uncertainty, making exploration inferences difficult. It is "not quite good but ok" for very basic unpredictability hints.  
* **2:** The text has mild uncertainty with 1-2 options, somewhat inconsistent. It is "ok" for simple exploration, and some might find it adequately unpredictable for basic contexts.  
* **3:** The text is "great" with moderate uncertainty (3-4 paths, some exploration emphasis), effectively presenting multi-path elements.  
* **4:** Exhibits "very good" high uncertainty with 5 paths and strong trial emphasis, precisely defined for adaptive agentic exploration.  
* **5:** Demonstrates "outstanding" mastery with >5 nuanced alternatives, flawlessly emphasizing profound exploration; representing the <1% of exceptional texts.

**5. Interactivity**  
Evaluate simulated entity interactions or dialogues (e.g., "Character A queries B, B replies and counters, leading to joint decision"). This develops implicit collaboration in agents.  
* **1:** The text has no interactions, with static descriptions requiring effort to infer any dynamics. It is "not quite good but ok" for very basic reciprocity hints.  
* **2:** The text shows simple one-way exchanges, often awkward. It is "ok" for basic interactions, and some might find it adequately collaborative for simple scenarios.  
* **3:** The text exhibits "great" moderate multi-turn reciprocity, effectively simulating entity dynamics.  
* **4:** Demonstrates "very good" dense, multi-entity reciprocal interactions with sophisticated structure, perfectly defining collaborative agentic elements.  
* **5:** Achieves "outstanding" mastery with profound, nuanced multi-turn dynamics, flawless in impact; representing the <1% of benchmark texts.

**6. Feedback Loops**  
Evaluate action-result-adjustment cycles (e.g., "Tried method A, failed due to error, adjusted to B, succeeded after refinement"). This builds implicit resilience in agents.  
* **1:** The text has no cycles, with static elements hindering loop inferences. It is "not quite good but ok" for very basic adjustment hints.  
* **2:** The text includes a single basic loop, somewhat imprecise. It is "ok" for simple feedback, and some might find it adequately iterative.  
* **3:** The text is "great" with 2 iterative loops, effectively presenting action-result-adjustment.  
* **4:** Exhibits "very good" multiple loops (3) with deep adjustments, precisely structured for resilient agentic learning.  
* **5:** Demonstrates "outstanding" mastery with >3 complex, nuanced loops, flawless in depth; representing the <1% of exceptional texts.

**7. Multi-Agent Collaboration**  
Evaluate cooperative multi-entity dynamics (e.g., "Group divides tasks: A scouts, B builds, they synchronize and adapt if issues arise"). This fosters implicit teamwork in agents.  
* **1:** The text has no collaboration, with solo elements. It is "not quite good but ok" for very basic coordination hints.  
* **2:** The text shows basic pairing, often inconsistent. It is "ok" for simple teamwork, and some might find it adequately collaborative.  
* **3:** The text exhibits "great" moderate coordination with roles, effectively implying multi-agent synergy.  
* **4:** Demonstrates "very good" complex multi-party synergy with adjustments, precisely defined for team dynamics.  
* **5:** Achieves "outstanding" mastery with profound, nuanced coordination, flawless in execution; representing the <1% of benchmark texts.

**8. Resource Management**  
Evaluate handling of limited resources (e.g., "Budget split: 40% to materials, 60% to labor, reallocate if shortfall"). This develops implicit optimization in agents.  
* **1:** The text has no resource mentions, with no allocation hints. It is "not quite good but ok" for very basic management.  
* **2:** The text includes basic handling, somewhat clunky. It is "ok" for simple trade-offs, and some might find it adequately optimizing.  
* **3:** The text is "great" with moderate trade-offs, effectively presenting resource dynamics.  
* **4:** Exhibits "very good" detailed, dynamic optimization, precisely structured for efficient allocation.  
* **5:** Demonstrates "outstanding" mastery with profound, nuanced management, flawless in impact; representing the <1% of exceptional texts.

**9. Risk Assessment and Trade-offs**  
Evaluate weighing risks and benefits (e.g., "High reward but 30% failure risk; trade-off by hedging with alternative plan"). This sharpens implicit prudence in agents.  
* **1:** The text has no risks, with no weighing elements. It is "not quite good but ok" for very basic assessment hints.  
* **2:** The text shows simple pros/cons, often imprecise. It is "ok" for basic trade-offs, and some might find it adequately prudent.  
* **3:** The text exhibits "great" multi-factor weighing, effectively implying balanced judgment.  
* **4:** Demonstrates "very good" deep, quantitative trade-offs, precisely defined for agentic caution.  
* **5:** Achieves "outstanding" mastery with profound, nuanced assessments, flawless in depth; representing the <1% of benchmark texts.

**10. Long-Term Planning Horizon**  
Evaluate extended multi-phase planning (e.g., "Phase 1: learn; Phase 2: apply; Phase 3: scale over years"). This cultivates implicit vision in agents.  
* **1:** The text is immediate-only, with no phases. It is "not quite good but ok" for very basic foresight hints.  
* **2:** The text has short-term 1 phase, somewhat vague. It is "ok" for simple planning, and some might find it adequately visionary.  
* **3:** The text is "great" with medium 2-3 phases, effectively extending horizons.  
* **4:** Exhibits "very good" long-term spanning 4 phases with contingencies, precisely structured.  
* **5:** Demonstrates "outstanding" mastery with >4 nuanced, cross-year phases, flawless in foresight; representing the <1% of exceptional texts.

**11. Tool Invocation Simulation**  
Evaluate simulated tool usage with details (e.g., "Invoke calculator: input 5+3, output 8; integrate into budget calc"). This mimics implicit tool-calling in agents.  
* **1:** The text has no tools, with no invocation hints. It is "not quite good but ok" for very basic simulation.  
* **2:** The text shows vague use, often inconsistent. It is "ok" for simple tools, and some might find it adequately integrative.  
* **3:** The text exhibits "great" detailed invocations with parameters, effectively mimicking agentic tools.  
* **4:** Demonstrates "very good" frequent, sequenced invocations, precisely defined for external integration.  
* **5:** Achieves "outstanding" mastery with profound, nuanced tool chains, flawless in simulation; representing the <1% of benchmark texts.

**12. Adaptation to Changes**  
Evaluate dynamic pivots to disruptions (e.g., "Plan A failed due to rain; adapt by switching to indoor alternative, refine further if needed"). This builds implicit flexibility in agents.  
* **1:** The text is static, with no adjustments. It is "not quite good but ok" for very basic resilience hints.  
* **2:** The text includes simple single adjustments, somewhat awkward. It is "ok" for basic pivots, and some might find it adequately flexible.  
* **3:** The text exhibits "great" moderate multi-change responses, effectively handling disruptions.  
* **4:** Demonstrates "very good" profound, layered adaptations to complex shifts, precisely structured.  
* **5:** Achieves "outstanding" mastery with exceptional, nuanced resilience, flawless in depth; representing the <1% of exceptional texts.
"""


def batch_inference(llm, sampling_params, prompts):
    messages = []
    for prompt in prompts:
        messages.append([{"role": "user", "content": prompt}])

    # Generate outputs
    outputs = llm.chat(
        messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": True},  # Set to False to strictly disable thinking
    )

    # Print the outputs.
    for output in outputs[:10]:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, \n\nGenerated text: {generated_text!r}")


def load_model():
    llm = LLM(model="Qwen/Qwen3-8B")
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=40000)

    with open("../local_data/test_data_0731/sample_100_each_data.json", "r") as f:
        data = json.load(f)

    prompts = []
    data = data[:100]
    for each in data:
        text = each["data"][0]["text"]
        prompt = prompt_template.replace("A `given_text` for evaluation.", text)
        prompts.append(prompt)
    return llm, sampling_params, prompts


def main():
    llm, sampling_params, prompts = load_model()
    batch_inference(llm, sampling_params, prompts)


if __name__ == "__main__":
    main()


