# Commonsense memory for the agent
# Written by Jiageng Mao & Junjie Ye

import subprocess
import json

class CommonSenseMemory:
    def __init__(self) -> None:
        self.common_sense = {
            "Traffic Rules": TRAFFIC_RULES,
        }

    def retrieve(self, knowledge_types: list = None) -> str:
        commonsense_prompt = "\n"
        if knowledge_types:
            for knowledge_type in knowledge_types:
                commonsense_prompt += f"{'*' * 5}{knowledge_type}:{'*' * 5}\n"
                for rule in self.common_sense[knowledge_type]:
                    commonsense_prompt += f"- {rule}\n"
        else:
            for knowledge_type, rules in self.common_sense.items():
                commonsense_prompt += f"{'*' * 5}{knowledge_type}:{'*' * 5}\n"
                for rule in rules:
                    commonsense_prompt += f"- {rule}\n"
        return commonsense_prompt

TRAFFIC_RULES = [
    "Avoid collision with other objects.",
    "Always drive on drivable regions.",
    "Avoid driving on occupied regions.",
    "Pay attention to your ego-states and historical trajectory when planning.",
    "Maintain a safe distance from the objects in front of you.",
    "When passing a gas station, be sure to slow down, avoid rapid acceleration or overtaking.",
    "Try to avoid following novice drivers, and maintain a safe distance.",
    "On speed-limited sections, you must follow the speed limit regulations and keep your speed within a reasonable range.",
    "When driving at night, avoid driving on white roads as much as possible, and if you must, be extremely cautious.",
    "When driving in the rain, reduce your speed and maintain a safe distance.",
    "When turning, do not exceed a speed of 30 km/h.",
    "When following a car, maintain a distance of at least one car length and frequently check your rearview mirror.",
    "When driving on special road sections such as slopes or bridges, maintain sufficient distance to avoid the need for emergency evasive actions.",
    "At night, try to drive in the middle lane to facilitate turning or avoiding obstacles when necessary.",
    "Upon entering a residential area, observe the surroundings carefully and avoid sudden acceleration or abrupt turns if there are children, elderly people, or obstacles ahead.",
    "When encountering buses, taxis, electric vehicles, or large trucks, maintain a safe distance to avoid close parallel driving.",
    "Do not change lanes frequently while driving; maintain a consistent speed.",
    "When approaching intersections or obstructions, slow down and be prepared to brake at any time.",
    "Do not frequently change lanes or make sharp turns at intersections; driving straight is the safest method.",
    "Yielding to pedestrians is a basic responsibility of every driver.",
    "Do not overtake at red light intersections to avoid potential danger.",
    "For vehicles traveling in the same direction, the left vehicle yields to the right; safety is the priority.",
    "In urban areas, avoid using high beams, turn off high beams when meeting other vehicles, and use high beams to alert the vehicle ahead in no-horn zones.",
    "Avoid driving alongside large trucks for extended periods; observe the truck's movements before overtaking to ensure safety.",
    "Do not overtake at intersections, especially avoid the large truck's right-side blind spot.",
    "Braking over speed bumps can damage the suspension system; slow down in advance and release the brake before crossing the bump.",
    "Always use the turn signal when changing lanes to inform other drivers of your intentions.",
    "Right turns must yield to left turns, and slow down when entering or exiting tunnels or making any turns.",
    "On highways, do not drive in the overtaking lane (leftmost lane) for extended periods.",
    "Turn on the turn signal in advance when turning or changing lanes.",
    "Adjust your headlights in advance when meeting other vehicles.",
    "Always slow down at intersections.",
    "Do not overtake recklessly when the large bus ahead slows down or stops.",
    "When the brake lights of the vehicle ahead are on, follow with light braking.",
    "Change to the appropriate lane in advance before making a turn.",
    "Do not drive alongside large vehicles for long periods.",
    "Maintaining a safe distance is crucial.",
]

# Function to call local model using curl
def call_local_model(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "qwen2:7b",
        "prompt": prompt,
        "stream": False
    }
    response = subprocess.run(
        ["curl", url, "-d", json.dumps(data), "-H", f"Content-Type: {headers['Content-Type']}"],
        capture_output=True,
        text=True
    )
    return response.stdout.strip()

################################################################################################################################

memory = CommonSenseMemory()

# Retrieve all traffic rules from memory
traffic_prompt = memory.retrieve(["Traffic Rules"])

# Modify prompt to ask for new traffic rules
enhanced_prompt = traffic_prompt + "\nPlease express traffic rules for automobiles using first-order logic expressions."

# Call local model with enhanced prompt to generate new traffic rules
generated_response = call_local_model(enhanced_prompt)

# Split the generated response into individual rules and replace TRAFFIC_RULES
generated_rules = [rule.strip() for rule in generated_response.split('\n') if rule.strip()]

print("Please express traffic rules for automobiles using first-order logic expressions.\n")

print("Generated Traffic Rules:\n")

for i, rule in enumerate(generated_rules, 1):
    print(f"{i}. {rule}")


