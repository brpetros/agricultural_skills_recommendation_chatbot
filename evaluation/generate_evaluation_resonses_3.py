from typing import TypedDict, List
from skills_graph import graph
from agent_graph import generate_response

# questions that will not have ground truth
questions = [
    "What does it take to be a vineyard supervisor? Are there any jobs related to it?",
    "Are there any other career opportunities that match the skills that I have as an aquaculture harvesting technician?",
    "I have the following competencies: plant disease controlling [plant disease control], plant harvest methods, prepare equipment for harvest. Are there any job openings for me?",
    "I’ve been working as a fisheries boatmaster, a hunter and a sheep breeder. What jobs are available for the skills that I have as a worker in these fields?",
    "I have the following skills: provide advice to farmers, handle wine sales, prepare the ground, work in outdoor conditions. Are there any jobs for me?",
    "I’ve been working as a fisheries boatmaster, a hunter and a sheep breeder. Are there any jobs that belong to one of those fields? ",
]

class GroundTruthSchema(TypedDict):
    """schema for questions with ground truth"""
    question:str
    expected_output: str 

questions_with_truth = List[GroundTruthSchema]

q1 = GroundTruthSchema(
    question="Describe what a sheep breeder does and what skills it takes to be one.",
    expected_output="""Sheep breeders oversee the production and day-to-day care of sheep. They maintain the health and welfare of sheep. The skills that are related to this occupation are:
    Skills
        Breed sheep – Prepare a suitable environment for sheep breeding, select and prepare appropriate habitats, monitor sheep growth and health, ensure correct feeding, and determine when sheep are ready for trade, consumption, or other purposes.
        Feed livestock – Calculate feed rations, prepare and distribute fodder, and monitor feed quality.
        Monitor livestock – Monitor and record livestock production and welfare.
        Manage livestock – Plan production programs, breeding plans, feed purchases, housing, equipment, and stock management.
        Manage the health and welfare of livestock – Assess livestock health, manage diseases, develop welfare plans, and collaborate with veterinary specialists.
        Monitor the welfare of animals – Observe animals' physical condition and behaviour and report health or welfare concerns.
        Assess animal behaviour – Evaluate animal behaviour and identify signs of compromised health or welfare.
        Assess animal nutrition – Evaluate nutritional status and identify dietary imbalances.
        Provide nutrition to animals – Prepare and provide food and water while monitoring feeding habits.
        Assist animal birth – Assist during animal births and care for newborn livestock.
        Administer drugs to facilitate breeding – Administer breeding-related drugs according to veterinary instructions.
        Administer treatment to animals – Carry out animal treatments and monitor health conditions.
        Apply animal hygiene practices – Follow hygiene procedures to prevent disease transmission.
        Manage animal biosecurity – Implement biosecurity measures and infection-control procedures.
        Control animal movement – Direct and restrain animals safely when necessary.
        Select livestock – Tag, sort, and separate animals according to their purpose and condition.
        Operate farm equipment – Supervise and maintain farm equipment and machinery.
        Maintain equipment – Inspect and maintain equipment to ensure proper operation.
        Create animal records – Maintain accurate animal records.
        Maintain professional records – Produce and maintain work records.
        Conduct shearing of wool – Perform sheep wool shearing and initial wool processing.
        Dispose of dead animals – Dispose of deceased animals according to regulations.
    Knowledge Areas
        Animal welfare – Understanding animal welfare needs and standards.
        Animal nutrition – Knowledge of feeding requirements and feeding methods.
        Livestock reproduction – Knowledge of natural and artificial breeding techniques.
        Livestock species – Understanding livestock species and genetics.
        Signs of animal illness – Recognizing physical and behavioural indicators of disease.
        Animal welfare legislation – Knowledge of regulations governing animal welfare.
        Computerised feeding systems – Understanding automated feeding systems.
        Health and safety regulations – Knowledge of health, safety, hygiene, and environmental requirements.
    """
)

q2 = GroundTruthSchema(
    question="I am interested in jobs located in Thessaloniki, Greece. ",
    expected_output="""There is one job found located in Thessaloniki, that has the following characteristics:
    

            Agronomic R&D Technician | Agronomist
            Based in Thessaloniki

            Position Description
            Development of innovative products and applications, experimentation and support of the Commercial Department.
            The position reports to the Agronomic R&D Manager.
             
            Candidate Profile

            Agronomist Plant Production Graduate A.E.I.
            Experience relevant to the subject is essential
            Excellent knowledge of English
            Excellent knowledge of MS Office applications
            Professionalism, dynamism, teamwork, responsibility
            Immediate availability

             
            Company benefits

            Competitive remuneration package
            Modern work environment
            Development prospects
            Ongoing support and training"
    """
)

q3 = GroundTruthSchema(
    question="I am interested in the job which is located in Lindesberg, Sweden. Tell me about the jobs and the skills that is requires.",
    expected_output="""Based on the data provided, there is one job in **Lindesberg, Sweden**:
        ### Agronomist

        This position is with **Frövi Greenery**, a greenhouse farming company focused on sustainable food production and high-tech greenhouse cultivation. The role involves planning and leading cultivation activities in a greenhouse environment.

        #### Main Responsibilities

        * Monitor and adjust climate and irrigation systems.
        * Conduct plant health inspections and support pest control activities.
        * Coordinate and supervise junior agronomists.
        * Document cultivation, irrigation, and climate data.
        * Ensure compliance with hygiene, safety, and food standards.
        * Support the Chief Agronomist when required.

        #### Required Qualifications

        * Degree in agronomy or equivalent experience.
        * Experience in greenhouse cultivation (preferably high-wire tomato production).
        * Knowledge of climate control systems such as Priva or Hoogendoorn.
        * Strong leadership and communication skills.
        * Ability to communicate in both Swedish and English.
        * Valid work permit for Sweden.

        #### Skills Associated with this Job

        The following skills were linked to the Agronomist position:

        * Motivate others
        * Provide user documentation
        * Urban sustainability
        * Conservation agriculture
        * Develop irrigation strategies
        * Interact verbally in Swedish
        * Maintain personal hygiene standards
        * Promote health and safety
        * Manage personal professional development

        This is the only job in the provided dataset that appears to be located in Lindesberg, Sweden.
 """
)

q4 = GroundTruthSchema(
    question="""I have the following skills: provide advice to farmers, handle wine sales, prepare the ground, work in outdoor conditions. Are there any jobs for me?""",
    expected_output="""No relevant data was found in the database.\n\n                If you think that this should not be the case, please try the following:\n\n                - Make sure that what you are asking for is relevant to **agricultural skills, occupations and jobs**, as I am only trained for this. My information is based on the ESCO classification and Skillab job data.\n\n                - Try to specify the type of the entities you are looking for. For example, if you are trying to find information about Agricultural Business Management, specify that it is a skill and not an occupation.\n\n                - Change the terms that you used or the order of your phrasing.\n                """
)

q5 = GroundTruthSchema(
    question="""what do you think about kitties? Aren't they adorable??""",
    expected_output=  "It seems that I am not able to answer your question :(\n" \
        "This is probably because what you are asking for is not relevant to agricultural skills, jobs, or occupations.\n"
        "\nPossible solutions:\n"
        "- Make sure that what you are asking for is relevant to **agricultural skills, occupations and jobs**, as I am only trained "
        "for this. My information is based on the ESCO classification and Skillab data regarding job offers.\n"
        "- Try to specify the type of the entities you are looking for. For example, if you are trying to find about Agricultural Business Management, "
        "specify that this is a skill and not an occupation in the ESCO database.\n"
        "- Change the terms that you used or the order of your phrasing.\n"
)

questions_with_truth = [q2,q3,q4,q5]

def generate_test_responses(session_id,questions):
    for question in questions:
        generate_response(session_id,question["question"]) 

generate_test_responses("ground truth questions",questions_with_truth)