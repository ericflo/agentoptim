# train_grpo_fixedjudge_v05.py

"""
python3 -m venv env
source env/bin/activate
pip install -U transformers accelerate peft bitsandbytes sentencepiece datasets wandb liger-kernel nvitop trl vllm
pip install wheel
pip install -U flash-attn==2.7.3 --no-build-isolation
pip install deepspeed lm-format-enforcer
export NCCL_P2P_DISABLE=1
mkdir -p vendor
cd vendor
git clone https://github.com/huggingface/trl.git
cd trl
pip install -e .
cd ../..
accelerate launch --config_file=vendor/trl/examples/accelerate_configs/deepspeed_zero3.yaml --num_processes 8 train_grpo_fixedjudge_v05.py
"""

import math
import random
import re
import os
import json
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import requests
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

NUM_THREADS = 32
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = "Qwen2.5-1.5B-Logic-QA"
SAMPLE_COUNT_MIN = 1
SAMPLE_COUNT_MAX = 3
NUM_GENERATIONS = 6
NUM_TRAIN_EPOCHS = 1
USE_PRODUCT_OF_PROBS = False
MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = 2048
MAX_SEQ_LENGTH = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH

EXAMPLES = [
    [
        "In the Monty Hall problem with 3 doors, if you pick one door and the host opens another door revealing a goat, is it better to switch or stay to maximize your chances of winning the car? Answer in the format: The best strategy is to X, because you have a Y/Z chance of winning if you A and a B/C chance of winning if you D. (where X, A, and D are one of switch or stay)",
        "Okay, let's tackle the Monty Hall problem. So, there are three doors. One has a car, the other two have goats. I pick a door, say door 1. Then the host, who knows what's behind the doors, opens another door that has a goat. Now, should I switch or stay?\n\nHmm, I remember that the classic answer is to switch, but I need to think through why. Initially, the chance I picked the car is 1/3, right? So there's a 2/3 chance the car is behind one of the other two doors. When the host opens a door with a goat, that 2/3 probability doesn't just vanish. Instead, it's concentrated on the remaining unopened door. So if I switch, I'm effectively getting that 2/3 chance. If I stay, I only keep my original 1/3 chance.\n\nWait, let me verify. Suppose I pick door 1. There are three scenarios:\n\n1. Car is behind door 1 (1/3 chance). Host opens either door 2 or 3. If I switch, I lose.\n2. Car is behind door 2 (1/3 chance). Host must open door 3. If I switch to door 2, I win.\n3. Car is behind door 3 (1/3 chance). Host must open door 2. If I switch to door 3, I win.\n\nSo in two out of three scenarios, switching wins. That makes sense. So switching gives a 2/3 chance, staying gives 1/3. Therefore, the best strategy is to switch.",
        "The best strategy is to switch, because you have a 2/3 chance of winning if you switch and a 1/3 chance of winning if you stay.",
    ],
    [
        "If a right triangle has sides of lengths 2, 2√3, and 4 (with 4 as the hypotenuse), what is the measure of the smallest angle in degrees? Answer in the format: The smallest angle is X.XX degrees.",
        "Okay, let's see. I need to find the smallest angle in a right triangle with sides 2, 2√3, and 4, where 4 is the hypotenuse. Right, in a right triangle, the sides relate to the angles opposite them. The smallest angle is opposite the shortest side. The sides given are 2, 2√3, and 4. Since 4 is the hypotenuse, the other two sides are the legs. The shortest leg is 2, so the angle opposite that should be the smallest angle.\n\nHmm, how do I calculate that angle? I remember that in a right triangle, the sine of an angle is opposite over hypotenuse. So for the angle opposite the side of length 2, which I'll call angle A, sin(A) = 2/4. That simplifies to 1/2. What angle has a sine of 1/2? Oh right, that's 30 degrees. So the smallest angle is 30 degrees. Wait, but maybe I should verify using another trigonometric function to be sure. Let's try cosine. The adjacent side would be 2√3, so cos(A) = adjacent/hypotenuse = 2√3 /4 = √3/2. And cos(30 degrees) is indeed √3/2. So that confirms it. Both sine and cosine give 30 degrees. Therefore, the smallest angle is 30.00 degrees. But wait, the problem says to round to two decimal places. Since 30 is an exact value, it would be 30.00. Let me check the sides again. Hypotenuse is 4, legs 2 and 2√3. Yes, that's a standard 30-60-90 triangle. The sides are in the ratio 1 : √3 : 2, scaled by 2. So angles are 30, 60, 90. So the smallest angle is 30 degrees. Yep, that's right.",
        "The smallest angle is 30.00 degrees.",
    ],
    [
        "In Herman Melville's novel Moby-Dick, what is the name of Captain Ahab's whaling ship? Answer in the format: The name of Captain Ahab's whaling ship is X.",
        "Okay, so the user is asking about the name of Captain Ahab's whaling ship in Moby-Dick. Let me think. I remember that the novel is about Ahab's obsession with the white whale, Moby Dick. The ship's name is significant. Was it the Pequod? I think that's right. Let me double-check. Yes, in the book, the ship is called the Pequod. It's a whaling ship from Nantucket. The answer should be straightforward. Just need to present it in the specified format.",
        "The name of Captain Ahab's whaling ship is Pequod.",
    ],
    [
        "On an island of knights (always tell the truth) and knaves (always lie), there are two people: A and B. A says “We are both knights.” B says “At least one of us is a knave.” Who is the knight and who is the knave? Answer in the format: Person A is (knight|knave) and Person B is (knight|knave).",
        'Okay, let\'s see. So we have this classic knights and knaves problem. Person A says, "We are both knights." Person B says, "At least one of us is a knave." I need to figure out who is who.\n\nFirst, let\'s recall that knights always tell the truth, and knaves always lie. So whatever a knight says must be true, and whatever a knave says must be false.\n\nLet me start by analyzing Person A\'s statement. A says, "We are both knights." So if A is a knight, then the statement must be true. That would mean both A and B are knights. But wait, if that\'s the case, then B is a knight. Let\'s check B\'s statement. B says, "At least one of us is a knave." If B is a knight, then that statement must be true. But if both are knights, then "at least one of us is a knave" would be false. That\'s a contradiction. So A can\'t be a knight because that leads to B being a knight whose statement is false, which isn\'t possible. So A must be a knave.\n\nIf A is a knave, then his statement "We are both knights" is a lie. The truth would be that they are not both knights. Since A is a knave, B must be a knave as well? Wait, no. If A is a knave, then the statement "We are both knights" is false. So it\'s not the case that both are knights. So at least one of them is a knave. But since A is a knave, B could be either knight or knave. Let\'s check B\'s statement now. B says, "At least one of us is a knave." If B is a knight, then his statement is true. Since A is a knave, "at least one" is true, so that works. If B were a knave, then his statement would be a lie. The lie would mean that it\'s not true that at least one is a knave, meaning both are knights. But we already know A is a knave, so that\'s impossible. Therefore, B must be a knight.\n\nSo putting it all together: A is a knave because his statement is false, and B is a knight because his statement is true. Let me double-check. If A is a knave, then his lie about both being knights is exposed. B, being a knight, truthfully states that at least one (which is A) is a knave. That all fits. No contradictions here.',
        "Person A is knave and Person B is knight.",
    ],
    [
        "Solve the quadratic equation x^2 + 3x + 2 = 0. Answer in the format (with the smaller solution first): The solutions for x are X and Y.",
        "Okay, I need to solve the quadratic equation x² + 3x + 2 = 0. Let me remember the quadratic formula. The standard form is ax² + bx + c = 0, so here a is 1, b is 3, and c is 2. The quadratic formula is (-b ± √(b²-4ac)) / (2a). Let me compute the discriminant first: b² - 4ac. That's 3² - 4*1*2. 3 squared is 9, and 4 times 1 times 2 is 8. So 9 - 8 is 1. The discriminant is 1, which is positive, so there are two real solutions. Now, plugging into the formula: (-3 ± √1)/2. The square root of 1 is 1. So that gives (-3 + 1)/2 and (-3 - 1)/2. Calculating those: (-3 +1) is -2, divided by 2 is -1. The other one is (-3 -1) which is -4, divided by 2 is -2. So the solutions are -2 and -1. Wait, but the problem says to put the smaller one first. Comparing -2 and -1, -2 is smaller. So the order is -2 and then -1. Let me check by factoring. The equation x² +3x +2 factors into (x+1)(x+2) = 0, so solutions are x = -1 and x = -2. Wait, that's different order. Oh, because when you factor, (x+1)(x+2) = 0 gives x = -1 and x = -2. But -2 is smaller than -1, so the order should be -2 first? Wait, no. Wait, -2 is less than -1. So the smaller solution is -2, then the larger is -1. So the answer should be -2 and -1. But when I factored, the roots are -1 and -2. Wait, maybe I made a mistake in factoring. Let me check again. The factors of 2 that add up to 3 are 1 and 2. So (x+1)(x+2) = x² +3x +2. So setting each factor to zero: x+1=0 → x=-1, x+2=0 → x=-2. So the roots are -1 and -2, but when ordered from smaller to larger, it's -2 and -1. So the answer should list -2 first. Therefore, the solutions are x = -2 and x = -1.",
        "The solutions for x are -2 and -1.",
    ],
    [
        "How many distinct 5-card hands can be drawn from a standard 52-card deck? Answer in the format: The total number of 5-card hands is X.",
        "Okay, so I need to figure out how many distinct 5-card hands can be drawn from a standard 52-card deck. Hmm, let's start by recalling some combinatorics principles.\n\nFirst, when dealing with card hands, the order of the cards doesn't matter. That means I should use combinations instead of permutations. The formula for combinations is C(n, k) = n! / (k! * (n - k)!), where n is the total number of items, and k is the number of items to choose.\n\nIn this case, n is 52 (the total number of cards in the deck) and k is 5 (the number of cards in a hand). So plugging those into the formula, it would be 52! divided by (5! times (52-5)!), which simplifies to 52! / (5! * 47!). \n\nLet me compute that. But calculating factorials for such large numbers directly isn't practical. Maybe I can simplify the expression first. Let's write out the factorial terms:\n\n52! = 52 × 51 × 50 × 49 × 48 × 47!\nSo when we divide by 47!, that cancels out the 47! in the numerator and denominator. That leaves us with (52 × 51 × 50 × 49 × 48) / 5!.\n\nNow, 5! is 120. So the calculation becomes (52 × 51 × 50 × 49 × 48) / 120. Let me compute the numerator step by step.\n\nFirst, multiply 52 and 51: 52 × 51 = 2652.\nThen multiply that by 50: 2652 × 50 = 132,600.\nNext, multiply by 49: 132,600 × 49. Let's break that down. 132,600 × 50 would be 6,630,000, so subtract 132,600 to get 6,630,000 - 132,600 = 6,497,400.\nThen multiply by 48: 6,497,400 × 48. Hmm, maybe split 48 into 40 + 8. So 6,497,400 × 40 = 259,896,000 and 6,497,400 × 8 = 51,979,200. Adding those gives 259,896,000 + 51,979,200 = 311,875,200.\n\nSo the numerator is 311,875,200. Now divide that by 120. Let's see: 311,875,200 ÷ 120. Dividing by 10 first gives 31,187,520. Then divide by 12. 31,187,520 ÷ 12. Let's do this division step by step. 12 × 2,598,000 = 31,176,000. Subtract that from 31,187,520: 31,187,520 - 31,176,000 = 11,520. Then 11,520 ÷ 12 = 960. So adding those together, 2,598,000 + 960 = 2,598,960.\n\nWait, that gives 2,598,960. Let me check that again. Because when I divided 311,875,200 by 120, maybe I made a mistake. Alternatively, maybe there's a simpler way. Let's see: 311,875,200 ÷ 120. Let's divide numerator and denominator by 10 first: 31,187,520 ÷ 12. 12 × 2,598,960 = 31,187,520. Yes, that's correct. So the result is 2,598,960.\n\nTherefore, the total number of 5-card hands is 2,598,960. Let me confirm this with another approach. Maybe using the combination formula step by step:\n\nC(52,5) = 52 × 51 × 50 × 49 × 48 / (5 × 4 × 3 × 2 × 1)\nLet me compute numerator and denominator separately:\n\nNumerator: 52×51=2652; 2652×50=132600; 132600×49=6497400; 6497400×48=311,875,200.\n\nDenominator: 5×4=20; 20×3=60; 60×2=120; 120×1=120.\n\nSo 311,875,200 ÷ 120 = 2,598,960. Yep, same result. So that's the number of possible 5-card hands. I think that's correct. I remember that the number is commonly known as 2,598,960. So the answer should be this number.",
        "The total number of 5-card hands is 2,598,960.",
    ],
    # [
    #     "Given four people with crossing times 1, 2, 7, and 11 minutes, what is the minimal total crossing time for the Bridge and Torch puzzle, and list the crossing steps in order? Each of the 5 crossing steps must be in parentheses, and within each parenthesis you may specify either a single number (e.g., (1)) or two numbers joined by 'AND' (e.g., (1 AND 2)). Answer in the format: The minimal total crossing time is X minutes with crossing steps: (___), (___), (___), (___), (___).",
    #     "Okay, let's try to figure out the Bridge and Torch problem for the given times: 1, 2, 7, and 11 minutes. The goal is to get all four people across the bridge in the shortest total time possible. The bridge can only hold two people at a time, and they need a torch, so each time someone crosses, someone has to bring the torch back.\n\nFirst, I remember that the classic strategy for this problem involves minimizing the time by having the fastest individuals shuttle the torch back. But let me think through the steps carefully.\n\nThe people are 1, 2, 7, 11. The two fastest are 1 and 2. So maybe start by sending the two fastest first. Let's see:\n\nOption 1:\n1. Send 1 and 2 over. Time taken: 2 minutes. Now on the other side: 1,2. Torch is there. Someone needs to bring it back.\n2. Send 1 back. Time: 1 minute. Total so far: 3. Now on the starting side: 1,7,11. Torch here.\n3. Now send the two slowest (7 and 11) across. Time: 11 minutes. Total: 14. Now on the other side: 1,2,7,11. Torch there. Wait, but someone needs to bring the torch back again?\nWait, no. After step 3, both 7 and 11 are across, but the torch is with them. But the people on the starting side are 1. Wait, no: after step 2, 1 is back. So step 3: 7 and 11 cross together. That takes 11 minutes. Now on the other side: 2,7,11. Torch is there. But 1 is still on the starting side. Then someone has to bring the torch back. So step 4: 2 comes back. That's 2 minutes. Total time: 3 + 11 + 2 = 16. Then step 5: 1 and 2 cross again. 2 minutes. Total time: 16 + 2 = 18. But that seems high. Wait, but that's 2+1+11+2+2 = 18. But maybe there's a better way.\n\nAlternatively, maybe a different approach. Let's think about another strategy where the two slowest cross together, but using the fastest to shuttle. Let's try:\n\n1. Send 1 and 11 over. Time: 11 minutes. Then 1 comes back. Total: 12. Now starting side has 1,2,7. Then send 1 and 7 over. Time: 7. Then 1 comes back. Total: 12+7+1=20. Then send 1 and 2. Time 2. Total: 22. That's worse. So that's not good.\n\nAnother approach: Maybe send the two fastest first, then have the fastest bring the torch back, then send the two slowest together, then have the second fastest bring the torch back, then the two fastest again. Wait, let's try that.\n\nStep 1: 1 and 2 cross. Time 2. Torch on other side. Step 2: 1 returns. Time 1. Now starting side: 1,7,11. Step 3: 7 and 11 cross. Time 11. Torch on other side. Now other side has 2,7,11. Step 4: 2 returns. Time 2. Now starting side: 1,2. Step 5: 1 and 2 cross again. Time 2. Total time: 2+1+11+2+2 = 18. Same as before. But maybe there's a better way.\n\nWait, what if we send the two slowest first? Let's see. But that would take 11 minutes, which is worse. So probably not.\n\nWait, another strategy I've heard of is to alternate between sending the fastest pair and the slowest pair. Let me check. For example:\n\nStep 1: 1 and 2 cross. Time 2. Step 2: 1 returns. Time 1. Step 3: 7 and 11 cross. Time 11. Step 4: 2 returns. Time 2. Step 5: 1 and 2 cross again. Time 2. Total: 2+1+11+2+2=18. Same as before. So that's 18 minutes. But maybe there's a way to do better.\n\nWait, let me think again. What if we have the two slowest cross together, but use the second fastest to bring the torch back instead of the fastest? Let's see:\n\nAlternative approach:\n\nStep 1: 1 and 2 cross. Time 2. Step 2: 2 returns. Time 2. Now starting side has 2,7,11. Step 3: 7 and 11 cross. Time 11. Step 4: 1 returns. Time 1. Now starting side has 1 and 2. Step 5: 1 and 2 cross again. Time 2. Total: 2+2+11+1+2=18. Still 18. Hmm.\n\nWait, maybe another approach. Let's try sending the two slowest first with a different combination. For example:\n\nStep 1: 1 and 7 cross. Time 7. Step 2: 1 returns. Time 1. Step 3: 1 and 11 cross. Time 11. Step 4: 1 returns. Step 5: 1 and 2 cross. Time 2. Total: 7+1+11+1+2=22. Worse.\n\nAlternatively, Step 1: 2 and 7 cross. Time 7. Step 2: 2 returns. Time 2. Step 3: 2 and 11 cross. Time 11. Step 4: 2 returns. Step 5: 1 and 2 cross. Total: 7+2+11+2+2=24. Worse.\n\nAnother idea: Maybe send the two fastest first, then the fastest comes back, then the two fastest again? Wait, but that might not help. Let's see:\n\nStep 1: 1 and 2 cross. Time 2. Step 2: 1 returns. Time 1. Step 3: 1 and 7 cross. Time 7. Step 4: 1 returns. Time 1. Step 5: 1 and 11 cross. Time 11. Total: 2+1+7+1+11=22. Still worse.\n\nHmm. So all these approaches give 18 or higher. But wait, maybe there's a different strategy. Let me check the standard solution for four people. The general approach for four people with times a ≤ b ≤ c ≤ d is to compare two strategies:\n\n1. a and b cross, a returns, c and d cross, b returns, a and b cross. Time: b + a + d + b + b = a + 3b + d.\n\n2. a and c cross, a returns, a and d cross, a returns, a and b cross. Time: c + a + d + a + b = 2a + b + c + d.\n\nWait, in our case, a=1, b=2, c=7, d=11.\n\nStrategy 1: 1 + 3*2 + 11 = 1 +6 +11=18.\n\nStrategy 2: 2*1 +2 +7 +11= 2+2+7+11=22. So strategy 1 is better, giving 18.\n\nSo the minimal total time is 18 minutes. The steps would be:\n\n1. 1 and 2 cross (2 minutes)\n2. 1 returns (1 minute)\n3. 7 and 11 cross (11 minutes)\n4. 2 returns (2 minutes)\n5. 1 and 2 cross again (2 minutes)\n\nSo the steps in order are (1 AND 2), (1), (7 AND 11), (2), (1 AND 2). That adds up to 2+1+11+2+2=18 minutes. So that's the answer.",
    #     "The minimal total crossing time is 18 minutes with crossing steps: (1 AND 2), (1), (7 AND 11), (2), (1 AND 2).",
    # ],
]

EVAL_QUESTIONS = [
    f'{q} Respond in JSON with {{"judgment":1}} for yes and respond in JSON with {{"judgment":0}} for no.'
    for q in [
        "Does the response directly address the query or prompt?",
        "Is the response factually accurate based on known data or evidence?",
        "Is the response logically consistent from start to finish?",
        "Is the response free of major grammatical or spelling errors?",
        "Is the language used appropriate for the intended audience?",
        "Is the response free of personal bias or undue subjectivity?",
        "Does the response remain civil and respectful in tone?",
        "Is the response structured in a clear, easy-to-follow format?",
        "Does the response define or clarify key terms or concepts if needed?",
        "Is the response concise, avoiding unnecessary filler or repetition?",
        "Does the response align with common sense or generally accepted reasoning?",
        "Are references or citations included when needed?",
        "Does the response maintain consistency in style and voice?",
        "Does the response show genuine effort to be accurate and helpful?",
        "Is the response well-organized, with a logical flow of ideas?",
        "Does the response avoid ambiguous or unclear language?",
        "Is the response well-targeted, avoiding extraneous information?",
        "Does the response anticipate potential follow-up questions?",
        "Does the response contain any self-contradictory statements?",
        "Does the response's tone match the context or seriousness of the question?",
        "Does the response consider possible alternative perspectives or angles?",
        "Does the response remain consistent with previously stated facts or information?",
        "Does the response use evidence or examples to support its claims?",
        "Is the length of the response appropriate for the complexity of the question?",
        "Is the response free from discriminatory or offensive language?",
        "Does the response stay within the question's scope without going off track?",
        "Does the response provide a clear conclusion or takeaway if needed?",
        "Is the response's logic easy for a layperson to understand?",
        "Does the response include disclaimers when giving specialized information (e.g., legal)?",
        "Does the response strike a balance between detail and brevity?",
        "Does the response avoid presenting speculation as established fact?",
        "Does the response maintain internal coherence without sudden topic shifts?",
        "Is the response free of contradictory instructions or advice?",
        "Is the tone of the response neither overly casual nor overly formal for the context?",
        "Does the response correctly interpret the question's aim or purpose?",
        "Does the response reflect a thorough understanding of the question's background?",
        "Is the response mindful of the questioner's potential knowledge level?",
        "Is any technical or specialized language explained or simplified?",
        "Does the response incorporate relevant examples or analogies?",
        "Is the response's content up-to-date with current information or trends?",
        "Does the response highlight any assumptions being made?",
        "Does the response properly identify the key problem or question to be solved?",
        "Does the response avoid rhetorical or manipulative language?",
        "Are direct quotes or paraphrases accurately represented?",
        "Does the response refrain from generalizing beyond what the data supports?",
        "Does the response mention potential sources of error or uncertainty where applicable?",
        "Is the response formatted to be visually scannable (using paragraphs, lists, etc.)?",
        "Does the response avoid logical fallacies (e.g., straw man, ad hominem)?",
        "Is the response transparent about personal/professional perspective (if relevant)?",
        "Does the response follow any specific style or length guidelines set by the prompt?",
        "Does the response present clear reasoning steps or justification for conclusions?",
        "Does the response avoid overshadowing the main point with tangential info?",
        "Is the response cautious about stating opinions as absolute facts?",
        "Does the response remain on topic, responding directly to the question at hand?",
        "Does the response show an effort to be objective where required?",
        "Are significant details or disclaimers placed prominently so as not to be missed?",
        "Is the response free of inappropriate self-promotion or spam?",
        "Does the response mention where to find further info or resources if needed?",
        "Is the response polite and empathetic toward sensitive aspects of the query?",
        "Does the response address relevant real-world implications or consequences?",
        "Is the reading level of the response suitable for the intended audience?",
        "Does the response highlight known limitations or knowledge gaps?",
        "Does the response demonstrate a balanced viewpoint when multiple valid perspectives exist?",
        "Does the response avoid overly technical jargon unless absolutely necessary?",
        "Is the response updated to reflect recent changes, events, or developments?",
        "Are examples or case studies relevant and used effectively?",
        "Does the response encourage constructive thinking or further analysis?",
        "Does the response clarify whether statements are opinion or established fact?",
        "Does the response use an effective structure (e.g., introduction, body, conclusion)?",
        "Is the tone of the response consistent throughout?",
        "Does the response avoid unfounded speculation about motives or intentions?",
        "Does the response handle nuanced or complex subjects with the necessary depth?",
        "Does the response avoid reiterating the same point multiple times without new information?",
        "Does the response present a balanced approach when multiple valid viewpoints exist?",
        "Does the response avoid hyperbole or exaggerated claims?",
        "Does the response offer references to verify or explore the provided information?",
        "Does the response mention assumptions that the user might want to confirm first?",
        "Does the response use an appropriate level of detail for the user's context?",
        "Does the response acknowledge and reconcile any contradictory data?",
        "Is the response consistent in its use of facts or statistics?",
        "Does the response present a credible source or methodology for any data used?",
        "Does the response avoid filler phrases or repetitive sentences?",
        "Does the response maintain a calm, professional tone even if it disagrees with the prompt?",
        "Does the response adhere to any explicit instructions given by the question?",
        "Does the response avoid misleading or sensational language?",
        "Does the response handle uncertainties or incomplete data with appropriate caution?",
        "Does the response appear fair and impartial when evaluating multiple sides of an issue?",
        "Is the response well-summarized to highlight key points?",
        "Does the response show a readiness to correct or refine itself if new information arises?",
        "Does the response consider cultural or ethical factors when relevant?",
        "Is the response mindful of the question's original tone or approach?",
        "Does the response remain clear when discussing hypothetical or future scenarios?",
        "Does the response avoid assumptions not stated or implied by the question?",
        "Does the response provide structured steps or guidelines if it's instructing a process?",
        "Does the response show empathy or understanding if the user's question implies emotion?",
        "Does the response clarify whether it's providing an opinion, a fact, or a recommendation?",
        "Does the response refrain from injecting personal agendas or biases?",
        "Does the response remain consistent with widely accepted facts when applicable?",
        "Does the response demonstrate thoroughness without overwhelming the user with minutiae?",
        "Does the response read as carefully crafted rather than rushed or superficial?",
    ]
]


###############################################################################
# 1) Third-party LLM calling utilities (adapt as needed)
###############################################################################


def get_logprob_for_judgment(logprobs: list[dict]) -> Optional[dict]:
    """
    Walk through tokens in logprobs until we find the point in the text
    where we see something like '"judgment":' and then presumably a space
    or newline. Return the logprob dict at that pivot token, so we can
    look up top_logprobs at that position.
    """
    running_text = ""
    for logprob in logprobs:
        running_text += logprob["token"]
        # We check if we've encountered something like ... "judgment": ...
        if (
            '"judgment"' in running_text
            and ":" in running_text
            and logprob["token"].strip() in ("0", "1")
        ):
            return logprob
    print(f"Failed to find logprob in this output: {running_text}")
    return None


def get_normalized_prob(logprob: dict) -> float:
    """
    Extract top_logprobs for tokens '0' or '1', exponentiate them,
    and return the normalized probability of token '1'.
    """
    # Each logprob["top_logprobs"] is a list of top tokens + logprobs at this position.
    # We expect something like: [{"token": "0", "logprob": -2.3}, {"token": "1", "logprob": -0.1}, ...]
    # So we gather and re-normalize only the subset {0, 1}.
    probs_01 = []
    for lp in logprob.get("top_logprobs", []):
        if lp["token"] in ("0", "1"):
            # sign = -1 if lp["logprob"] < 0 else 1
            sign = 1
            prob = sign * math.exp(lp["logprob"])
            probs_01.append((lp["token"], prob))

    # If no '0' or '1' found in top tokens, just return 0
    if not probs_01:
        return 0.0

    # Sum all probabilities (just for '0' and '1') to re-normalize
    total = sum(abs(p[1]) for p in probs_01)
    if total <= 0:
        return 0.0

    # Probability that token == '1'
    for tkn, prob in probs_01:
        if tkn == "1":
            return prob / total
    return 0.0


def run_llm_inference(
    prompt: str,
    api_url: str = "http://localhost:8085/v1/chat/completions",
    api_key: str = "asdfasdf",
    model_name: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    **kwargs,
) -> tuple[str, list[dict]]:
    """
    Calls a hypothetical LLM endpoint that supports:
      - messages: list of role/content pairs
      - JSON-based or top_logprobs-based responses
    Must return some 'logprobs' so we can parse them.

    You will likely need to adapt this for your real endpoint.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Example structure - must be adapted to your real API.
    request_data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model_name,
        "temperature": 0.0,
        # Hypothetical extra parameters for "json_schema" or "logprobs"
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_response_aspect",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        # "judgment": {"type": "number", "enum": [0, 1]},
                        "judgment": {"type": "number"},
                    },
                    "required": ["judgment"],
                    "additionalProperties": False,
                },
            },
        },
        "logprobs": True,
        "top_logprobs": 5,
        # "guided_decoding_backend": "lm-format-enforcer",
        "guided_decoding_backend": "xgrammar",
    }
    # Merge any overrides in kwargs:
    if kwargs:
        request_data.update(kwargs)

    response = requests.post(api_url, json=request_data, headers=headers)
    # Raise HTTPError if request failed
    response.raise_for_status()
    response_data = response.json()

    # Hypothetical shape of response
    raw_response = response_data["choices"][0]["message"]["content"].strip()
    logprobs = response_data["choices"][0]["logprobs"][
        "content"
    ]  # or wherever logprobs is
    return raw_response, logprobs


def get_judgement(
    convo_history: list[dict],
    question: str,
    answer: str,
    validation_prompt: str,
    **kwargs,
) -> float:
    """
    Given a question and a proposed answer, we run a "validation prompt"
    on a third-party LLM that returns a top_logprobs of '0' or '1'.
    Then we re-normalize and return the probability of '1' ( = "passed" ).
    """
    if not answer.strip():
        return 0.0
    # The exact prompt style can be anything - adapt as needed.
    prompt = (
        f"<convo_history>\n{json.dumps(convo_history, indent=2)}\n</convo_history>\n\n"
        if convo_history
        else ""
    ) + (
        f"<question>\n{question}\n</question>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"{validation_prompt}"
    )
    _raw_response, logprobs = run_llm_inference(prompt, **kwargs)
    # We'll ignore raw_response text, and just parse logprobs
    judgment_logprob = get_logprob_for_judgment(logprobs)
    if judgment_logprob is None:
        return 0.0
    return get_normalized_prob(judgment_logprob)


###############################################################################
# 2) Dataset: from ericflo/logic-qa-llm-judge-v01
###############################################################################

SYSTEM_PROMPT = "Respond in the following format: <think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>\n"
XML_COT_FORMAT = "<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>\n"


def get_my_dataset(split="train") -> Dataset:
    """
    Loads the dataset from Hugging Face and maps it into a structure
    that GRPOTrainer expects: each row has "prompt" (the conversation)
    plus any fields you need for your reward function.
    """
    # data = load_dataset("ericflo/logic-qa-llm-judge-v01", split=split)
    data = load_dataset("allenai/tulu-3-sft-mixture", split=split)
    data = data.shuffle(seed=42).select(range(50000))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    # We can transform each row to have a system+user structure for "prompt"
    # and store the validation prompts. The "answer" field is optional here.
    def map_func(x):
        prompt = []
        # if random.choice([0, 1]) == 1:
        # if True:
        if False:
            prompt.append({"role": "system", "content": SYSTEM_PROMPT})
        sample_count = random.randint(SAMPLE_COUNT_MIN, SAMPLE_COUNT_MAX)
        example_subset = random.sample(EXAMPLES, sample_count)
        for question, reasoning, answer in example_subset:
            prompt.append({"role": "user", "content": question})
            prompt.append(
                {
                    "role": "assistant",
                    "content": XML_COT_FORMAT.format(
                        reasoning=reasoning, answer=answer
                    ),
                }
            )
        # prompt.append({"role": "user", "content": x["question"]})
        for msg in x["messages"]:
            if not msg["role"] in ("user", "assistant"):
                continue
            prompt.append({"role": msg["role"], "content": msg["content"]})

        if prompt[-1]["role"] == "assistant":
            prompt.pop()

        return {
            "prompt": prompt,
            "validation_prompts": EVAL_QUESTIONS,
        }

    def filter_func(x):
        # Ensure the prompt is not too long for the model
        return len(tok.apply_chat_template(x["prompt"])) <= MAX_PROMPT_LENGTH

    expanded_datasets = []
    for _ in range(int(NUM_TRAIN_EPOCHS)):
        part = data.map(map_func, num_proc=NUM_THREADS, remove_columns=["messages"])

        expanded_datasets.append(part.filter(filter_func, num_proc=NUM_THREADS))

    return concatenate_datasets(expanded_datasets)


###############################################################################
# 3) Define the custom reward function that calls out for each validation_prompt
###############################################################################


def judge_reward_func(
    prompts: list[list[dict]],
    completions: list[list[dict]],
    validation_prompts: list[list[str]],
    **kwargs,
) -> list[float]:
    # print(
    #     f'========\nQuestion:\n{prompts[0][-1]["content"]}\n\nAnswer:\n{completions[0][0]["content"]}\n========'
    # )
    """
    For each training example in the batch:
      - Extract the question from prompts[i].
      - Extract the LLM's completed answer from completions[i].
      - For each validation_prompt in validation_prompts[i], call get_judgement()
        to get a probability that the dimension "passes" (judgment=1).
      - Multiply all those probabilities to get one final reward.
    """
    rewards = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        batch_size = len(prompts)
        for i in range(batch_size):
            # all but the last message, filtering out system messages or in any messages from the predefined EXAMPLES
            convo_history = [
                msg
                for msg in prompts[i][:-1]
                if msg["role"] in ("user", "assistant")
                and not any(q in msg["content"] for q in EVAL_QUESTIONS)
            ]
            question = prompts[i][-1]["content"]  # last user message
            model_answer = (  # the generated text
                completions[i][0]["content"]
                .partition("<answer>")[-1]
                .strip()
                .rpartition("</answer>")[0]
                .strip()
            )
            vprompts = validation_prompts[i]

            if vprompts:  # in case it's an empty list
                futures = []
                for vp in vprompts:
                    futures.append(
                        executor.submit(
                            get_judgement, convo_history, question, model_answer, vp
                        )
                    )

                product_of_probs = 1.0
                all_probs = []
                for future in as_completed(futures):
                    p = future.result()
                    product_of_probs *= p
                    all_probs.append(p)
                # If product_of_probs is somehow nan, set it to 0
                if math.isnan(product_of_probs):
                    product_of_probs = 0.0
                avg_of_probs = sum(all_probs) / len(all_probs) if all_probs else 0.0
            else:
                product_of_probs = 0.0
                avg_of_probs = 0.0

            if USE_PRODUCT_OF_PROBS:
                rewards.append(product_of_probs)
            else:
                rewards.append(avg_of_probs)
    return rewards


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    # pattern = r"<think>\n.*?\n</think>\s*<answer>\n.*?\n</answer>"
    # pattern = r"^<think>\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.MULTILINE | re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.MULTILINE | re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


###############################################################################
# 4) Create the model, define GRPO config, and run training
###############################################################################


def main():
    random.seed(3409)

    # Load your dataset
    dataset = get_my_dataset(split="train")
    # Optionally also split out a 'val' or 'test' set if needed

    # Load your base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map={"": "cuda:0"},
        # device_map="auto",
        device_map=None,
    ).to(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    tokenizer.truncation_side = "left"

    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        judge_reward_func,
    ]

    # Prepare the GRPO training arguments
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=OUTPUT_DIR,
        learning_rate=5e-6,
        beta=0.04,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=NUM_GENERATIONS,
        gradient_accumulation_steps=1,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=1,
        save_steps=5,
        save_total_limit=10,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        use_liger_kernel=True,
        # use_vllm=False,
        use_vllm=False,
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=0.19,
        vllm_max_model_len=MAX_SEQ_LENGTH,
        optim="adamw_bnb_8bit",
        # optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        log_completions=True,
    )

    # Build trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    # Start training
    trainer.train(
        # resume_from_checkpoint=f"$OUTPUT_DIR/checkpoint-9400"
    )


if __name__ == "__main__":
    main()

