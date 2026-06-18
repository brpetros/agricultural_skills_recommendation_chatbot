from agent_graph import generate_response
from typing import TypedDict, List
from log_data import Interaction
import json



def generate_test_responses(session_id,skill):
    output_1 = generate_response(session_id,f"I need information about the following skill: {skill}") # response and metadata are saved in the jsonl new_interactions
    print(f"output for skill {skill}: {output_1}")
    output_2 = generate_response(session_id,"What occupations require this skill?")
    print(output_2)
    output_3 = generate_response(session_id,"I am interested in the first occupation you mentioned. Are there any jobs that contain it? ")
    print(output_3)
    output_4 = generate_response(session_id,"Are there any other skills that this occupation requires?")
    print(output_4)
    
def generate_multiple_responses(skills_to_evaluate):
    for skill in skills_to_evaluate:
        generate_test_responses(f"eval_test for {skill}",skill)

def generate_test_responses_1(session_id,occupation):
    output_1 = generate_response(session_id,f"I am a {occupation}. I wish to change my career. Tell me what paths I could follow with the skills that are essential for what I do.") # response and metadata are saved in the jsonl new_interactions
    print(f"output for occupation {occupation}: {output_1}")
    output_2 = generate_response(session_id,"In order to pursue the last path that you proposed, what are the important skills that I should focus on?")
    print(output_2)
    output_3 = generate_response(session_id,"I think I can manage this. Is there any job relevant to this occupation?")
    print(output_3)
    
    
def generate_multiple_response_occs(occupations_to_evaluate):
    for occupation in occupations_to_evaluate:
        generate_test_responses(f"occupation testing for {occupation}",occupation)



# queries not based on actual context
queries_occupations_related = [
    "What specific occupations are defined by a professional's ability to coordinate large field crews, assign daily tasks, and schedule field shifts?",
    "Which occupations require a deep mastery of financial planning, crop export logistics, and high-level corporate farming strategy?",
    "What occupations are centered around analyzing soil chemistry, mapping nutrient deficiencies, and managing vineyards?",
    "If an individual specializes in maximizing outputs in hydroponic greenhouses while minimizing chemical runoff, what occupations fit their profile?",
    "What occupations are typically held by experts who supervise swine breeding operations, evaluate animal health traits, and manage herd data?",
    "Which professional roles are dedicated entirely to streamlined workflow planning and raw material processing setups inside a facility?",
    "What career paths are rooted in teaching sustainable farming systems that protect biodiversity and leverage natural ecosystems?",
    "Which job profiles expect a professional to act as a scientific expert on plant health, seed rotation matrices, and biological crop sciences?",
    "What positions focus heavily on the daily operation of automated milking systems and diagnostic field sensors?",
    "What roles focus purely on land grading, fertilizing soil beds, and preparing fields before seeding takes place?",
    "Which careers match a professional who spends their career handling post-harvest cold-chain logistics and food product quality control?",
    "Which job roles require advanced expertise in mechanical troubleshooting, sensor calibration, and routine upkeep of modern automated combine harvesters?",
    "What standard field positions involve the physical labor of gathering mature yields and operating mechanical harvesters?",
    "Which careers grant a professional complete operational control over a commercial orchard, from planting schedules to disease mitigation?",
    "What career opportunities are available for individuals transitioning into equestrian stable management and animal welfare administration?",
    "Which roles are defined by the primary duty of judging the genetic potential of breeding cattle?",
    "What trade careers require an individual to be bilingual or multilingual as a core competency?",
    "What government or regulatory positions are responsible for conducting biosecurity audits on commercial poultry facilities?",
    "Which careers focus on developing drought-resistant seed strains to boost overall crop volumes in laboratory settings?",
]

queries_skills_related = [
    "What competences are required to work as a Logistics Manager, and are there any relevant open positions available?",
    "What does it take to become a Warehouse Order Picker, and are there currently any matching job openings?",
    "Which capabilities should I have to work as a Warehouse Supervisor, and are there any relevant vacancies available?",
    "What qualifications are expected from a professional who teaches horse riding techniques, and are there any related job opportunities at the moment?",
    "What skills are needed to work as a Sports Instructor, and are there any open positions relevant to this occupation?",
    "What requirements are important for a specialist responsible for procurement categories and supplier markets, and are there any suitable vacancies available?",
    "What competences are necessary for a professional guiding ships through busy harbours and waterways, and are there any related job openings currently available?",
    "What abilities are needed to work as a Solar Power Plant Operator, and are there any matching vacancies at the moment?",
    "What knowledge and skills are expected from an ICT Security Technician, and are there any relevant open positions available?",
    "What does it take to work as a Public Procurement Specialist, and are there any current job opportunities for this role?",
    "What competences are necessary for a professional leading crop production activities on a farm, and are there any related vacancies available?",
    "What skills should I develop to manage day-to-day agricultural operations and personnel, and are there any matching job openings currently available?",
    "What qualifications are needed for someone providing technical support in agricultural production, and are there any relevant positions available right now?",
    "What capabilities are expected from a professional involved in the development and implementation of agricultural policies, and are there any suitable vacancies available?",
    "What competences are required for a worker responsible for the care and management of farm animals, and are there any open positions related to this occupation?",
    "What requirements are associated with a career focused on crop cultivation and agronomic practices, and are there any relevant job opportunities available?",
    "What skills would I need to work as an Agronom, and are there any current vacancies that match this profession?",
    "What competences are expected from a Technical Sales Representative in the agricultural sector, and are there any open positions available?",
    "What qualifications are required for the role of Operations Manager at an agricultural enterprise, and are there any relevant job openings currently available?",
    "What does it take to work as a Greenhouse Worker at a food technology company, and are there any matching vacancies available?"
]

# queries based on existing data combinations - to test the context retrieval process - saved at test interactions 3
queries_job_related = [
    "Are there any job openings for a Logistics Manager responsible for coordinating warehouse operations and product distribution?",
    "What vacancies are available for professionals working as Warehouse Supervisors overseeing staff and inventory activities?",
    "Show me job opportunities for a Warehouse Order Picker involved in preparing and dispatching customer orders.",
    "Are there any open positions for a Public Procurement Specialist managing purchasing procedures and supplier contracts?",
    "What jobs are available for an ICT Security Technician responsible for maintaining cybersecurity systems and infrastructure?",
    "I am looking for vacancies related to operating and monitoring solar power generation facilities.",
    "Show me openings for professionals who provide technical sales support for agricultural products and services.",
    "What positions are currently available for individuals managing large-scale crop production activities on farms?",
    "Are there any job opportunities for agricultural technicians supporting farming operations and production processes?",
    "I want to find vacancies for professionals responsible for supervising livestock care, feeding, and animal welfare.",
    "Show me listings for workers involved in greenhouse cultivation and controlled-environment food production.",
    "What open positions exist for professionals responsible for planning and managing day-to-day agricultural operations?",
    "Are there any vacancies for specialists who evaluate supplier markets and manage procurement categories?",
    "I am searching for job opportunities related to guiding vessels safely through harbours and restricted waterways.",
    "What roles are available for professionals who teach horse riding techniques and supervise equestrian training activities?",
    "Show me openings for sports instructors who coach participants and promote physical activity programs.",
    "Are there any positions available for professionals involved in agricultural policy development and implementation?",
    "I would like to see vacancies for agronomy professionals specializing in crop cultivation and production improvement.",
    "What job opportunities are available for candidates experienced in warehouse logistics, inventory control, and distribution planning?",
    "Are there any open roles in Sweden for professionals responsible for coordinating farm personnel, field activities, and agricultural resources?"
]



def generate_single_interactions(session_id,queries):
    outputs = []
    for query in queries:
        outputs.append(generate_response(session_id,query))
    return outputs


# list of convesations
test_conversations_jobs = [
    [
        "What are the required skills for the position Administrators and advisors with a focus on farm animals, located in Östersund, Sweden?",
        "What occupations are related to this position?",
        "I am interested in the first occupation you mentioned. What skills should I have for that occupation?"
    ],
    [
        "Which occupations are associated with the job Administrators and advisors with a focus on farm animals in Östersund, Sweden?",
        "What skills are required for this position?",
        "Can you tell me more about the last skill you mentioned?"
    ],
    [
        "What skills are required for the position Chef till funktionen Jordbrukarstöd in Östersund, Sweden?",
        "Which occupations are related to this job?",
        "Tell me more about the first occupation you listed.",
        "What additional skills would help me succeed in that occupation?"
    ],
    [
        "What occupations are related to Säsongsarbetare till Vattenenheten located in Östersund, Sweden?",
        "What skills are required for this role?",
        "Can you explain the first skill you mentioned in more detail?"
    ],
    [
        "What skills should I have for the position AGRARBETRIEBSWIRTIN/AGRARBETRIEBSWIRT (M/W/D) in Freising, Germany?",
        "What occupations are related to this position?",
        "I would like to know more about the last occupation you mentioned."
    ],
    [
        "Which occupations are connected to the job Agraringenieur/ Anbauberater (m/w/d) in Egeln, Germany?",
        "What skills are needed for this role?",
        "What does the first occupation you listed typically do?",
        "What skills are especially important for that occupation?"
    ],
    [
        "What skills are required for the position Agricultural adviser to the Councilors in Sjuhärad in Tranemo, Sweden?",
        "Which occupations are related to this position?",
        "Tell me more about the first occupation you mentioned."
    ],
    [
        "What occupations are associated with Agricultural adviser to the Councilors in Sjuhärad located in Tranemo, Sweden?",
        "What skills are needed for the job itself?",
        "Can you explain the last skill you mentioned?"
    ],
    [
        "What skills are required for Agricultural adviser to the Councilors in Sjuhärad in Tranemo, Sweden?",
        "What occupations are related to it?",
        "What does the second occupation you mentioned involve?",
        "Which skills are most important for that occupation?"
    ],
    [
        "What skills should I have for the position Agricultural mechanic M/F in Louhans, France?",
        "Which occupations are related to this position?",
        "I am interested in the first skill you mentioned. Could you explain it further?"
    ],
    [
        "What occupations are related to the job Agricultural Technician M/F in Cognac, France?",
        "What skills are required for this position?",
        "Tell me more about the first occupation you listed."
    ],
    [
        "What skills are required for the position Agronom, Forslundagymnasiet located in Umeå, Sweden?",
        "Which occupations are related to this role?",
        "What skills would I need for the last occupation you mentioned?"
    ],
    [
        "Which occupations are associated with the position Application expert in Umeå, Sweden?",
        "What skills are required for this job?",
        "Could you explain the first skill in more detail?",
        "Is that skill useful in any of the occupations you mentioned?"
    ],
    [
        "What skills should I have for the role Försökstekniker in Umeå, Sweden?",
        "What occupations are related to this role?",
        "Tell me more about the first occupation."
    ],
    [
        "What occupations are related to the position Greenhouse assistant in Umeå, Sweden?",
        "What skills are needed for this position?",
        "Can you explain the last skill you mentioned?"
    ],
    [
        "What skills are required for the position Stable technician in Umeå, Sweden?",
        "Which occupations are related to it?",
        "I am interested in the first occupation you listed. What does it do?",
        "What skills are most important for that occupation?"
    ],
    [
        "Which occupations are associated with the role Stable technician located in Umeå, Sweden?",
        "What skills are required for the job itself?",
        "Tell me more about the first skill you mentioned."
    ],
    [
        "What skills are required for the position Agronomic R&D Researcher | Γεωπόνος in Larisa, Greece?",
        "What occupations are related to this position?",
        "What skills would I need for the first occupation you mentioned?"
    ],
    [
        "What occupations are associated with Agronomic R&D Specialist | Γεωπόνος - Λάρισα in Larisa, Greece?",
        "What skills are required for this role?",
        "Can you tell me more about the last occupation you mentioned?",
        "What skills are especially valuable for that occupation?"
    ],
    [
        "What skills are required for the position Farm Veterinarian in Larisa, Greece?",
        "Which occupations are related to this role?",
        "I am interested in the first occupation you mentioned. What skills should I have for that occupation?",
        "Can you explain one of those skills in more detail?"
    ]
]

test_conversations_occupations = [
    [
        "Are there any jobs available for an agricultural scientist?",
        "What competences are required for that occupation?",
        "Are there any other occupations that require knowledge of environmental protection and sustainability regulations?",
        "Which of those occupations also have job openings available?"
    ],
    [
        "What vacancies exist for professionals who research soil, plants, and animals to improve agricultural processes?",
        "What does it take to qualify for such a role?",
        "Are there any other occupations that require expertise in agricultural production?",
        "Can you show me relevant job opportunities for those occupations as well?"
    ],
    [
        "Are there any openings for a crop production worker?",
        "What skills are needed to perform that job successfully?",
        "Which other occupations require practical agronomical crop production experience?"
    ],
    [
        "I am interested in jobs involving practical crop cultivation and farming activities. Are there any available?",
        "What requirements are expected from candidates for those positions?",
        "What other occupations make use of crop production knowledge?",
        "Do any of them currently have vacancies?"
    ],
    [
        "Are there any job opportunities for a land-based machinery supervisor?",
        "What competences are necessary for that profession?",
        "Which other occupations require the ability to organise machinery services and coordinate operations?"
    ],
    [
        "Show me vacancies for professionals responsible for planning agricultural machinery services.",
        "What skills would I need to obtain one of these positions?",
        "Are there any related occupations that also require client coordination and operational planning?",
        "What jobs are available for those occupations?"
    ],
    [
        "Are there any openings for a livestock worker?",
        "What does it take to work in that occupation?",
        "Which other occupations require expertise in animal health and welfare?"
    ],
    [
        "I would like to work with breeding, feeding, and caring for animals. Are there any suitable jobs available?",
        "What qualifications or competences are needed for these roles?",
        "Are there any other occupations that require animal production and welfare skills?",
        "Do they have any current vacancies?"
    ],
    [
        "Are there any job offers for a writer?",
        "What skills are important for becoming successful in that role?",
        "Which other occupations require proofreading text as a competence?"
    ],
    [
        "I am looking for vacancies related to writing books, novels, or other literary works.",
        "What requirements should I meet to qualify for such positions?",
        "Are there other occupations that require strong grammar and spelling abilities?",
        "Can you show me relevant job opportunities?"
    ],
    [
        "Are there any jobs available for an Agricultural Policy Officer?",
        "What does it take to get this role?",
        "Are there any other occupations that require the last competence you mentioned?"
    ],
    [
        "I'm interested in occupations related to agricultural sustainability. What role would fit?",
        "Which skills are essential for that occupation?",
        "Are there any open positions for that occupation?",
        "What other occupations use the last skill you mentioned?"
    ],
    [
        "Are there any jobs available for a Project Manager?",
        "What skills are needed to become a Project Manager?",
        "Which other occupations require the last competence you mentioned?"
    ],
    [
        "What occupation focuses on coordinating resources and achieving business goals?",
        "What does this role require?",
        "Are there any job openings for that occupation?",
        "What other occupations use the last skill you mentioned?"
    ],
    [
        "Are there any jobs available for an Activity Leader?",
        "What skills are important for this occupation?",
        "Which other occupations require the last competence you mentioned?"
    ],
    [
        "I enjoy organizing events and leading groups. Which occupation suits me?",
        "What skills would I need?",
        "Are there any vacancies for that occupation?",
        "What other occupations use the last skill you mentioned?"
    ],
    [
        "Are there any jobs available for a Crisis Situation Social Worker?",
        "What skills are needed for this occupation?",
        "Which other occupations require the last competence you mentioned?"
    ],
    [
        "Which occupation helps people during emergencies and difficult life situations?",
        "What skills are essential?",
        "Are there any open positions related to that occupation?",
        "Which other occupations use the last skill you mentioned?"
    ],
    [
        "Are there any jobs available for an Energy Systems Engineer?",
        "What skills are required for this role?",
        "What other occupations require the last competence you mentioned?"
    ],
    [
        "I want to work on sustainable energy solutions. Which occupation would fit me?",
        "What skills should I develop?",
        "Are there any job opportunities for that occupation?",
        "What other occupations require the last skill you mentioned?"
    ]
]

test_conversations_skills = [
  [
    "Are there any jobs available for someone who is able to cooperate with colleagues in order to ensure that operations run effectively?",
    "What specific role titles could a person apply for if they have this competence?",
    "Are there any other occupations, such as warehouse worker or logistics manager, that require this exact skill?",
    "What is the full occupation description provided for an activity leader in this context?"
  ],
  [
    "I am looking for roles that require the skill manage crop production. Which positions in the database fit this criteria?",
    "Could you list all five occupations associated with this agricultural management skill?",
    "What are the responsibilities described for an agronomist under these roles?"
  ],
  [
    "Which job openings match a candidate who can promote the inclusion of agricultural programmes on a local and national level to acquire support for sustainability awareness?",
    "Are there any policy-related occupations linked to this specific responsibility?",
    "How does the file differentiate the occupation description of a general policy officer from an agricultural policy officer?"
  ],
  [
    "What job title is listed for someone capable of providing information to clients about the role of physical activity and the importance of healthy activities for daily living?",
    "Which health, fitness, or care occupations are connected to this healthy lifestyle promotion skill?",
    "Can you provide the detailed occupation description for a shiatsu practitioner or a pilates teacher from the data?"
  ],
  [
    "If someone specializes in animal production science, including animal nutrition, husbandry, and bio-security, what jobs can they find?",
    "What are the target occupations for a professional with this background?",
    "What does a veterinary medicine lecturer or an animal behaviourist do according to the occupation descriptions?"
  ],
  [
    "Can you find jobs like Business manager for the new center for sustainable primary production or Driftmaskinist based on their required skills?",
    "What is the core skill description shared by these roles and other positions like Agronom, Forslundagymnasiet?",
    "Which occupations, like pesticides sprayer or aquaculture harvesting worker, are expected to practice this skill?"
  ],
  [
    "I want to know which jobs are suitable for someone who can perform crop production duties such as planning, tilling, planting, and controlling pests.",
    "If I pursue these options, what occupation labels will I be classified under?",
    "What does the data describe as the main function of an agronomic crop production team leader?"
  ],
  [
    "Is the job Jordbruksverket söker säsongsanställda till växtskyddscentralerna! or Technical Sales Representative associated with promoting agricultural policies?",
    "What other jobs share this exact skill requirement?",
    "What are the specific occupations that someone holding this skill would be qualified to do?"
  ],
  [
    "Which unique job title only appears under the skill label promote healthy lifestyle?",
    "What types of alternative or complementary therapist occupations are listed under this specific job profile?",
    "How does the system describe the role of a home care aide or a leisure attendant?"
  ],
  [
    "What positions, such as Herdenmanager m/w/d or Land and Agricultural Estate Operations Director M/F, require expertise in herd health management and animal husbandry?",
    "What are the five distinct animal-related occupations associated with these positions?",
    "Can you give me the exact description for the live animal transporter or animal care attendant occupations?"
  ],
  [
    "Are there any jobs available for someone who is knowledgeable in quality assurance procedures?",
    "Which roles like Växthusarbetare till food tech-bolag can a person apply for with this background?",
    "What other occupations besides aircraft engine specialist require an understanding of these procedures?",
    "Could you provide the occupation description for a boat rigger or fiberglass laminator found under this skill?"
  ],
  [
    "I am looking for jobs that involve the ability to develop pension schemes. What roles are present in the dataset?",
    "What are the precise occupations that handle these specific retirement packages?",
    "How does the file describe the functional duties of a pension scheme manager compared to a pensions administrator?"
  ],
  [
    "Which employment positions are open to individuals specializing in the food and beverage industry?",
    "What core professional occupations, such as industrial cook or baker, are linked to this sector?",
    "What specific tasks are outlined for a food production engineer or master coffee roaster according to the data?"
  ],
  [
    "What roles can I find if my focus is on corporate sustainability?",
    "Are there consultancy or analysis occupations related to this expertise listed in the file?",
    "Can you provide the description for an environmental policy officer or an energy analyst under this topic?"
  ],
  [
    "Which job openings match a candidate skilled to manage transport of goods from warehouse facilities?",
    "What logistical or storage management occupations are associated with these specific transport duties?",
    "What are the precise responsibilities of a warehouse order picker or warehouse manager according to the source file?"
  ],
  [
    "If a candidate wants to apply for the position Agraringenieur/ Anbauberater (m/w/d), what diverse skills must they possess according to the data?",
    "Which across-the-board occupations like import export specialist or energy consultant align with this job's required skill sets?",
    "Is this specific job listing linked to both corporate sustainability and quality assurance procedures?"
  ],
  [
    "Can you find any seasonal or specialized listings like Växthusarbetare till food tech-bolag (deltid) in the records?",
    "What is the primary skill label that governs this specific food tech position?",
    "What unexpected heavy industrial occupations, such as ammunition assembler or waterway construction labourer, share this same quality control standard?"
  ],
  [
    "Are there any listings like Technical assistant (m/f/d) breeding garden available in the database?",
    "What strategy-focused skill label is this technical assistant role mapped to?",
    "What are the target occupation labels for someone holding this financial/administrative competence?"
  ],
  [
    "Which distinct job entry is shared precisely between corporate sustainability and managing crop production environments like Gårdsansvarig?",
    "What are the environmental defense occupations tied directly to this sustainability focus?",
    "How does the text define the occupational role of a green ICT consultant?"
  ],
  [
    "What logistics-driven career paths can someone pursue if they are qualified to oversee supply chain operations from warehouse facilities?",
    "What are the day-to-day duties listed for a standard warehouse worker under these options?",
    "Which management occupation takes overall responsibility for storage facilities and the personnel working inside them?"
  ]
]



def generate_conversational_interactions(session_id_start, conversations):
    for i, conversation in enumerate(conversations,1):
        for query in conversation:
            output = generate_response(f"{session_id_start}: {i}",query)
            if output.startswith("It seems that I am not able to answer your question") or output.startswith("No relevant data was found in the database"):
                print(f"no answer at conversation {i}")
                break 
    return True


    
