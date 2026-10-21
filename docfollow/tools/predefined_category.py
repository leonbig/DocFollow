"""
Predefined category module containing entity types and relation definitions.
This module centralizes all taxonomy definitions for consistent usage across the project.

该模块支持五个不同文本领域的特定实体分类：
- 学术与工程（Academic & Engineering）
- 金融与商业（Finance & Business）
- 教育（Education）
- 文学与艺术（Literature & Art）
- 政治与法律（Politics & Law）
"""

from typing import Dict, List, Any, Optional

# 句子类型划分
# Refined sentence relation definitions in English, with precise interpretation and keyword cues
RELATION_DEFINITIONS: Dict[str, str] = {
    "Causality": "B expresses a causal link with A, either giving a reason (cause) or stating a result (effect). Keywords: because, therefore, leads to, results in.",
    "Condition": "B specifies a prerequisite or necessary condition for A to occur or hold. Keywords: if, unless, provided that.",
    "Purpose": "B states the intended goal or purpose underlying A. Keywords: to, in order to, so that.",
    "Comparison": "B draws a similarity or contrast between A and another entity or idea. Keywords: like, unlike, similarly, whereas, compared to.",
    "Expansion": "B provides additional context, background, or elaboration on A without changing its core meaning.",
    "Rewriting": "B paraphrases or reformulates A using different wording, summarizing or restating its semantic content.",
    "Summary": "B provides a concise overview or generalization of A's main content. Keywords: in summary, to sum up, overall.",
    "Opinion": "B offers a subjective judgment, evaluation, interpretation, or recommendation about A.",
    "Sequence": "B indicates the chronological or logical order relating to A, marking progression. Keywords: first, next, then, subsequent.",
    "Transition": "B signals a shift to a new topic or perspective without elaborating on A. Keywords: now, moving on, regarding.",
    "Definition": "B defines or clarifies the meaning of a term or concept from A. Keywords: is defined as, means, refers to.",
    "Question": "B poses a question or inquiry about A. Keywords: why, what, how, does, is?",
    "Exception": "B introduces an exception or limitation to the assertion in A. Keywords: except that, apart from, aside from.",
    "Others": "B relates to A in a way that doesn't clearly fit any of the other defined relationship categories. This is a catch-all category for relationships that are ambiguous or unique."
}

# List of logical relation types for easy access
LOGICAL_RELATIONS = list(RELATION_DEFINITIONS.keys())


#############################################
# 基于领域的实体分类定义
#############################################

# 所有领域共有的实体类别
COMMON_ENTITY_TYPES = [
    {
        "category": "PER",
        "definition": "Person: specific individuals such as scientists, business leaders, educators, artists, or politicians."
    },
    {
        "category": "ORG",
        "definition": "Organization: institutional entities, including companies, universities, schools, museums, and government bodies."
    },
    {
        "category": "LOC",
        "definition": "Location: geographic places, such as countries, cities, regions, and natural features."
    },
    {
        "category": "FAC",
        "definition": "Facility: man-made structures or venues, such as airports, buildings, and landmarks."
    },
    {
        "category": "EVT",
        "definition": "Event: historical or planned occurrences, such as conferences, festivals, or historical incidents."
    },
    {
        "category": "DTM",
        "definition": "Date/Time: specific dates, time periods, or historical eras relevant to any domain."
    },
    {
        "category": "QTY",
        "definition": "Quantity/Measure: measurements with physical units applicable to any field."
    },
    {
        "category": "NUM",
        "definition": "Number/Identifier: unitless numbers, codes, or identifiers used across domains."
    },
    {
        "category": "CNP",
        "definition": "Concept/Object: other concrete objects or abstract concepts not covered elsewhere."
    },
    {
        "category": "OTHERS",
        "definition": "Others: entities that do not clearly fit into any other defined category. This is a catch-all category for ambiguous or unique entities."
    }
]

# 学术与工程领域的特有实体类别
ACADEMIC_ENGINEERING_ENTITY_TYPES = [
    {
        "category": "TECH",
        "definition": "Technology: specific technological innovations, systems, or methodologies such as quantum computing, blockchain, or AI algorithms; includes both established and emerging technologies."
    },
    {
        "category": "SCIENTIFIC_METHOD",
        "definition": "Scientific Method: specific research methodologies, experimental designs, analytical approaches, or systematic procedures used in scientific inquiry; includes methodological frameworks unique to different scientific disciplines."
    },
    {
        "category": "THEORY",
        "definition": "Theory: scientific theories, principles, laws, hypotheses, or academic frameworks that explain phenomena or guide research; includes both established and emerging theoretical constructs."
    },
    {
        "category": "ACADEMIC_INSTITUTION",
        "definition": "Academic Institution: research centers, laboratories, academic departments, or specialized scientific facilities dedicated to advancing knowledge in specific disciplines; distinct from general educational organizations."
    },
    {
        "category": "INNOVATION_OUTCOME",
        "definition": "Innovation Outcome: patents, scientific discoveries, research findings, technological breakthroughs, or engineering solutions that represent significant contributions to their fields; includes both tangible and conceptual innovations."
    }
]

# 金融与商业领域的特有实体类别
FINANCE_BUSINESS_ENTITY_TYPES = [
    {
        "category": "PRO",
        "definition": "Product: commercial goods, services, or intellectual properties offered to consumers or businesses; includes physical products, digital assets, and service offerings with market value."
    },
    {
        "category": "BUSINESS_MODEL",
        "definition": "Business Model: frameworks, strategies, or approaches that companies use to create, deliver, and capture value in economic, social, or cultural contexts; includes revenue models, operational structures, and market approaches."
    },
    {
        "category": "MARKET",
        "definition": "Market: economic systems, exchanges, trading platforms, or business sectors where commercial transactions occur; includes both physical marketplaces and virtual trading environments."
    },
    {
        "category": "FIN_INSTRUMENT",
        "definition": "Financial Instrument: tradable assets, contractual agreements, or monetary tools used for financial transactions; includes stocks, bonds, derivatives, cryptocurrencies, and other investment vehicles."
    },
    {
        "category": "ECONOMIC_INDICATOR",
        "definition": "Economic Indicator: statistical metrics, indices, or data points that reveal the economic performance of a market, sector, or economy; includes GDP, inflation rates, employment figures, and market indices."
    }
]

# 教育领域的特有实体类别
EDUCATION_ENTITY_TYPES = [
    {
        "category": "EDU_PROG",
        "definition": "Educational Program: structured learning initiatives, courses, curricula, or educational frameworks designed for knowledge transfer; includes both traditional and innovative educational offerings."
    },
    {
        "category": "LEARNING_OUTCOME",
        "definition": "Learning Outcome: measurable achievements, competencies, skills, or knowledge that learners are expected to demonstrate upon completion of educational experiences; includes cognitive, affective, and psychomotor domains."
    },
    {
        "category": "ASSESSMENT_METHOD",
        "definition": "Assessment Method: techniques, instruments, or approaches used to evaluate student learning or educational effectiveness; includes formative and summative assessments, standardized tests, and alternative evaluation methods."
    },
    {
        "category": "EDU_ROLE",
        "definition": "Educational Role: specialized positions, professional identities, or functional responsibilities within educational contexts; includes teachers, administrators, counselors, specialists, and support personnel."
    },
    {
        "category": "PEDAGOGY",
        "definition": "Pedagogy: teaching methodologies, learning theories, instructional designs, or educational approaches; encompasses both traditional and innovative ways of facilitating learning."
    }
]

# 文学与艺术领域的特有实体类别
LITERATURE_ART_ENTITY_TYPES = [
    {
        "category": "WOA",
        "definition": "Work of Art: creative productions or artistic expressions in any medium, including literature, visual arts, music, film, or performance; encompasses both classical and contemporary creative works."
    },
    {
        "category": "CREATIVE_ELEMENT",
        "definition": "Creative Element: specific components, motifs, symbols, themes, or stylistic features that constitute or characterize artistic works; includes recurring narrative elements, visual motifs, or symbolic representations."
    },
    {
        "category": "ART_MOVEMENT",
        "definition": "Artistic Movement: historical, cultural, or stylistic trends that influenced artistic expression during specific periods; encompasses schools of thought that shaped artistic development across various media."
    },
    {
        "category": "CULTURAL_CONTEXT",
        "definition": "Cultural Context: societal, historical, or intellectual environments that inform the creation, interpretation, or reception of artistic works; includes cultural periods, social movements, or geographical influences on art."
    },
    {
        "category": "ARTISTIC_MEDIUM",
        "definition": "Artistic Medium: materials, techniques, platforms, or channels through which creative expression is realized; includes traditional media like paint or film, as well as digital or experimental media forms."
    }
]

# 政治与法律领域的特有实体类别
POLITICS_LAW_ENTITY_TYPES = [
    {
        "category": "LAW",
        "definition": "Law: codified rules, regulations, statutes, treaties, or legislation that govern societal conduct; includes constitutional, criminal, civil, and international legal frameworks."
    },
    {
        "category": "LEGAL_PROCESS",
        "definition": "Legal Process: procedural mechanisms, protocols, or systems used to administer justice or resolve disputes; includes litigation procedures, alternative dispute resolution methods, and enforcement processes."
    },
    {
        "category": "POLITICAL_SYSTEM",
        "definition": "Political System: structural frameworks, governance models, or institutional arrangements through which political power is organized and exercised; includes democracies, autocracies, federations, and other government structures."
    },
    {
        "category": "POLICY",
        "definition": "Policy: formal courses of action, programs, initiatives, or strategies adopted by governments or institutions; includes domestic and foreign policies designed to address specific societal issues."
    },
    {
        "category": "CIVIC_RIGHT",
        "definition": "Civic Right: legally protected entitlements, freedoms, or guarantees afforded to individuals or groups within political societies; includes civil liberties, human rights, constitutional protections, and political privileges."
    }
]

# 为每个领域组合公共实体类别和特定领域实体类别
ACADEMIC_ENGINEERING_ENTITIES = COMMON_ENTITY_TYPES + ACADEMIC_ENGINEERING_ENTITY_TYPES
FINANCE_BUSINESS_ENTITIES = COMMON_ENTITY_TYPES + FINANCE_BUSINESS_ENTITY_TYPES
EDUCATION_ENTITIES = COMMON_ENTITY_TYPES + EDUCATION_ENTITY_TYPES
LITERATURE_ART_ENTITIES = COMMON_ENTITY_TYPES + LITERATURE_ART_ENTITY_TYPES
POLITICS_LAW_ENTITIES = COMMON_ENTITY_TYPES + POLITICS_LAW_ENTITY_TYPES

# 域到实体类别的映射
DOMAIN_TO_ENTITY_TYPES = {
    "academic_engineering": ACADEMIC_ENGINEERING_ENTITIES,
    "finance_business": FINANCE_BUSINESS_ENTITIES,
    "education": EDUCATION_ENTITIES,
    "literature_art": LITERATURE_ART_ENTITIES,
    "politics_law": POLITICS_LAW_ENTITIES
}

def get_entity_types_by_domain(domain: str) -> List[Dict[str, str]]:

    if domain not in DOMAIN_TO_ENTITY_TYPES:
        valid_domains = list(DOMAIN_TO_ENTITY_TYPES.keys())
        raise ValueError(f"Invalid domain: {domain}. Valid domains are: {valid_domains}")
    
    return DOMAIN_TO_ENTITY_TYPES[domain]


#############################################
# 基于领域的实体关系分类
#############################################

# 通用实体关系定义（适用于所有领域）
# General entity relation definitions (applicable to all domains)
GENERAL_ENTITY_RELATION_TYPES = [
    {
        "category": "Structural",
        "definition": "Entity A is structurally related to Entity B through composition, containment, or organization. Example: [Chapter] <-> [Book] (part-whole), [Department] <-> [Organization] (unit-organization), [Member] <-> [Committee] (element-collective). Focuses on inherent structural relationships rather than transient locations."
    },
    {
        "category": "Spatial",
        "definition": "Entity A is physically or geographically related to Entity B. Example: [Building] <-> [City] (located in), [Mountain] <-> [Valley] (adjacent to), [Star] <-> [Galaxy] (situated within). Focuses on physical positioning rather than administrative boundaries."
    },
    {
        "category": "Temporal",
        "definition": "Entity A relates to Entity B through time sequences, periods, or temporal associations. Example: [Ancient Period] <-> [Modern Era] (precedes), [Event] <-> [Date] (occurs on), [Process] <-> [Duration] (spans). Focuses purely on temporal aspects rather than causal relationships."
    },
    {
        "category": "Functional",
        "definition": "Entity A serves a purpose, function, or utility for Entity B. Example: [Tool] <-> [Task] (facilitates), [Medicine] <-> [Disease] (treats), [Software] <-> [Business Process] (automates). Focuses on utility relationships independent of causal or transformative aspects."
    },
    {
        "category": "Causal",
        "definition": "Entity A directly causes, produces, prevents, or impacts Entity B. Example: [Rain] <-> [Flood] (causes), [Vaccine] <-> [Disease] (prevents), [Policy] <-> [Outcome] (influences). Focuses on direct causal linkages rather than enablement or correlation."
    },
    {
        "category": "Derivational",
        "definition": "Entity A is transformed into, derived from, or serves as raw material for Entity B. Example: [Data] <-> [Information] (processed into), [Draft] <-> [Final Document] (develops into), [Source Material] <-> [Product] (transformed into). Focuses on transformation processes rather than simple causation."
    },
    {
        "category": "Attributive",
        "definition": "Entity A possesses Entity B as a property, characteristic, or measurable attribute. Example: [Product] <-> [Quality] (has property), [Person] <-> [Age] (has attribute), [Material] <-> [Strength] (has characteristic). Focuses on intrinsic properties rather than subjective assessments."
    },
    {
        "category": "Hierarchical",
        "definition": "Entity A is related to Entity B through classificatory, taxonomic, or power hierarchies. Example: [Manager] <-> [Employee] (supervises), [Species] <-> [Genus] (belongs to classification), [Subclass] <-> [Class] (is type of). Focuses on vertical organizational relationships rather than horizontal structural ones."
    },
    {
        "category": "Associative",
        "definition": "Entity A is linked to Entity B through social, contractual, or institutional arrangements. Example: [Person] <-> [Organization] (is member of), [Party A] <-> [Party B] (has agreement with), [Student] <-> [University] (is enrolled in). Focuses on formalized associations rather than inherent structural or hierarchical relationships."
    },
    {
        "category": "Comparative",
        "definition": "Entity A is compared to Entity B through similarity, difference, or relative measurement. Example: [Alternative A] <-> [Alternative B] (contrasts with), [Benchmark] <-> [Sample] (serves as standard for), [Original] <-> [Copy] (is replicated by). Focuses on explicit comparison rather than mere co-occurrence."
    },
    {
        "category": "Others",
        "definition": "Entity A relates to Entity B in a way that doesn't clearly fit any of the other defined relationship categories. This is a catch-all category for relationships that are ambiguous, complex, or unique and cannot be accurately classified using the existing taxonomy."
    }
]

# 学术与工程领域的特有实体关系
# Academic & Engineering domain-specific entity relations
ACADEMIC_ENGINEERING_RELATION_TYPES = [
    {
        "category": "Epistemic",
        "definition": "Knowledge relationship where Entity A provides evidence for, contradicts, or advances understanding of Entity B. Example: [Experiment] <-> [Theory] (supports/refutes), [Research] <-> [Knowledge Gap] (addresses). Focuses on evidential relationships in knowledge production."
    },
    {
        "category": "Innovation-Application",
        "definition": "Entity A represents a novel solution or approach that is applied to Entity B. Example: [Algorithm] <-> [Problem Domain] (solution-challenge), [Technology] <-> [Industry] (innovation-implementation context). Focuses on novel implementations rather than standard applications."
    },
    {
        "category": "Specification",
        "definition": "Entity A defines requirements, parameters, or standards for Entity B. Example: [Technical Standard] <-> [Implementation] (defines-conforms to), [Requirements Document] <-> [Design] (specifies-fulfills). Focuses on normative guidelines rather than descriptive documentation."
    },
    {
        "category": "Expertise-Field",
        "definition": "Entity A possesses specialized knowledge or capability within domain B. Example: [Specialist] <-> [Discipline] (expert in-domain of expertise), [Research Group] <-> [Research Area] (specializes in-specialty area). Focuses on knowledge competency rather than general associations."
    },
    {
        "category": "Methodological",
        "definition": "Entity A provides a systematic approach for studying, analyzing, or developing Entity B. Example: [Scientific Method] <-> [Phenomenon] (investigative approach-subject of study), [Development Framework] <-> [System] (methodology-product). Focuses on procedural frameworks rather than specific techniques."
    },
    {
        "category": "Experimental",
        "definition": "Entity A is used to test, validate, or explore Entity B under controlled conditions. Example: [Test Case] <-> [Software] (verification-subject), [Laboratory Setup] <-> [Hypothesis] (testing environment-proposition). Focuses on empirical validation rather than theoretical explanation."
    },
    {
        "category": "Technical-Dependency",
        "definition": "Entity A requires, relies on, or is built upon Entity B. Example: [Application] <-> [Operating System] (runs on-supports), [Extension] <-> [Core Technology] (extends-is extended by). Focuses on technical prerequisites rather than organizational dependencies."
    },
    {
        "category": "Academic-Lineage",
        "definition": "Entity A derives from, builds upon, or is intellectually descended from Entity B. Example: [Current Theory] <-> [Foundational Theory] (derived from-foundation for), [Student] <-> [Mentor] (intellectual descendant-predecessor). Focuses on intellectual heritage rather than generic derivation."
    },
    {
        "category": "Scientific-Impact",
        "definition": "Entity A produces measurable effects, influences, or consequences on Entity B. Example: [Research Finding] <-> [Policy] (informs-is informed by), [Technology] <-> [Environment] (impacts-is impacted by). Focuses on substantiated influence rather than general causation."
    },
    {
        "category": "Engineering-Constraint",
        "definition": "Entity A limits, restricts, or defines boundaries for Entity B. Example: [Resource Limitation] <-> [Design Solution] (constrains-is constrained by), [Physical Law] <-> [Engineering Possibility] (limits-is limited by). Focuses on design limiting factors rather than general restrictions."
    }
]

# 金融与商业领域的特有实体关系
# Finance & Business domain-specific entity relations
FINANCE_BUSINESS_RELATION_TYPES = [
    {
        "category": "Investment-Return",
        "definition": "Entity A represents a financial investment that yields or seeks Entity B as a financial outcome. Example: [Venture Capital] <-> [Startup Growth] (investment-outcome), [Bond Purchase] <-> [Interest Income] (capital-yield). Focuses on financial inputs and their expected or actual returns."
    },
    {
        "category": "Market-Product",
        "definition": "Entity A is a commercial offering that targets or serves Entity B as its consumer base. Example: [Enterprise Software] <-> [Corporate Clients] (product-target market), [Luxury Goods] <-> [High-income Consumers] (offering-customer segment). Focuses on product-market fit rather than general supply-demand dynamics."
    },
    {
        "category": "Supply-Demand",
        "definition": "Entity A represents resource availability that influences or responds to Entity B as market need. Example: [Oil Production] <-> [Energy Prices] (supply-market response), [Skilled Labor] <-> [Wage Levels] (resource availability-economic demand). Focuses on economic equilibrium mechanisms rather than specific product targeting."
    },
    {
        "category": "Regulatory-Compliance",
        "definition": "Entity A is a business entity or practice subject to Entity B as a legal or industry standard. Example: [Bank] <-> [Capital Requirements] (regulated entity-regulatory constraint), [Pharmaceutical Product] <-> [Safety Testing] (business activity-compliance requirement). Focuses on formal oversight relationships rather than general legal frameworks."
    },
    {
        "category": "Brand-Perception",
        "definition": "Entity A is a commercial identity associated with Entity B as a market perception. Example: [Luxury Brand] <-> [Exclusivity Image] (commercial entity-market perception), [Corporate Identity] <-> [Trust Rating] (brand-reputation element). Focuses on psychological and market positioning aspects rather than functional attributes."
    },
    {
        "category": "Resource-Allocation",
        "definition": "Entity A represents business assets or capital directed toward Entity B as a business activity or investment. Example: [R&D Budget] <-> [Innovation Initiative] (resource-allocation target), [Human Capital] <-> [Strategic Project] (business resource-priority area). Focuses on internal distribution decisions rather than external market dynamics."
    },
    {
        "category": "Risk-Management",
        "definition": "Entity A is a financial or business threat addressed by Entity B as a mitigation strategy. Example: [Market Volatility] <-> [Hedging Strategy] (risk-mitigation approach), [Cybersecurity Threat] <-> [Insurance Coverage] (business risk-protection mechanism). Focuses on uncertainty handling rather than general business planning."
    },
    {
        "category": "Business-Alliance",
        "definition": "Entity A forms a strategic partnership with Entity B for mutual commercial advantage. Example: [Manufacturer] <-> [Distributor] (production partner-distribution partner), [Technology Provider] <-> [Content Creator] (technical partner-content partner). Focuses on formal collaborative relationships rather than competitive positioning."
    },
    {
        "category": "Competitive-Strategy",
        "definition": "Entity A employs a market approach that addresses or responds to Entity B as a competitive force. Example: [Price Leadership] <-> [Market Share Growth] (strategy-competitive outcome), [Product Differentiation] <-> [Industry Rivalry] (strategic approach-competitive context). Focuses on market competition dynamics rather than internal business operations."
    },
    {
        "category": "Value-Chain",
        "definition": "Entity A represents a business activity or process that contributes to Entity B in a value creation sequence. Example: [Raw Material Sourcing] <-> [Manufacturing] (upstream-midstream activity), [Logistics] <-> [Retail Distribution] (support function-primary activity). Focuses on sequential value addition rather than organizational structure."
    }
]

# 教育领域的特有实体关系
# Education domain-specific entity relations
EDUCATION_RELATION_TYPES = [
    {
        "category": "Pedagogical-Learning",
        "definition": "Entity A represents a teaching approach that facilitates Entity B as a learning process or outcome. Example: [Project-Based Learning] <-> [Critical Thinking] (instructional method-learning outcome), [Socratic Method] <-> [Analytical Reasoning] (teaching technique-cognitive development). Focuses on specific teaching-learning connections rather than general educational activities."
    },
    {
        "category": "Assessment-Measurement",
        "definition": "Entity A is an evaluation method that measures or documents Entity B as a learning achievement. Example: [Standardized Test] <-> [Subject Mastery] (assessment tool-measured outcome), [Portfolio Evaluation] <-> [Skills Development] (assessment method-competency evidence). Focuses on educational measurement rather than general performance evaluation."
    },
    {
        "category": "Curriculum-Objective",
        "definition": "Entity A is an educational content structure designed to achieve Entity B as a learning goal. Example: [Science Curriculum] <-> [Scientific Literacy] (educational content-learning objective), [Arts Program] <-> [Creative Expression] (curriculum area-educational aim). Focuses on intentional content design rather than incidental learning."
    },
    {
        "category": "Educational-Provision",
        "definition": "Entity A is an educational institution or organization that offers Entity B as a learning program. Example: [University] <-> [Degree Program] (educational provider-structured offering), [Training Center] <-> [Professional Certification] (institution-credential program). Focuses on formal educational delivery rather than informal learning contexts."
    },
    {
        "category": "Developmental-Progression",
        "definition": "Entity A represents a learner state or educational level that advances toward Entity B as a developmental milestone. Example: [Early Literacy] <-> [Reading Fluency] (developmental stage-educational attainment), [Novice Status] <-> [Expert Performance] (learning phase-proficiency level). Focuses on educational growth trajectories rather than general learning activities."
    },
    {
        "category": "Policy-Implementation",
        "definition": "Entity A is an educational directive or mandate that guides Entity B as an institutional practice. Example: [Inclusion Policy] <-> [Classroom Accommodation] (educational regulation-implementation practice), [National Standard] <-> [Curriculum Alignment] (policy framework-educational response). Focuses on educational governance rather than general administrative procedures."
    },
    {
        "category": "Technology-Enhanced-Learning",
        "definition": "Entity A is an educational technology that enables or transforms Entity B as a learning experience. Example: [Learning Management System] <-> [Online Collaboration] (digital platform-educational activity), [Simulation Software] <-> [Applied Practice] (technological tool-experiential learning). Focuses on purposeful educational technology integration rather than general technology use."
    },
    {
        "category": "Professional-Enhancement",
        "definition": "Entity A represents educator training or development that improves Entity B as a teaching capability. Example: [Instructional Coaching] <-> [Classroom Management] (professional development-teaching skill), [Subject Matter Training] <-> [Content Expertise] (educator learning-teaching capacity). Focuses on teacher growth rather than student development."
    },
    {
        "category": "Learning-Environment",
        "definition": "Entity A is an educational setting or context that influences Entity B as a learning process or outcome. Example: [Collaborative Space] <-> [Peer Learning] (physical environment-educational process), [Inclusive Classroom] <-> [Equitable Participation] (learning context-educational result). Focuses on environmental factors in education rather than instructional methods."
    },
    {
        "category": "Research-Practice",
        "definition": "Entity A represents educational research or theory that informs Entity B as a teaching approach. Example: [Cognitive Science] <-> [Memory Enhancement Strategies] (research field-instructional application), [Learning Theory] <-> [Instructional Design] (educational research-classroom practice). Focuses on evidence-based practice rather than traditional teaching methods."
    }
]

# 文学与艺术领域的特有实体关系
# Literature & Art domain-specific entity relations
LITERATURE_ART_RELATION_TYPES = [
    {
        "category": "Creative-Genesis",
        "definition": "Entity A serves as inspiration or catalyst for Entity B as an artistic creation. Example: [Personal Experience] <-> [Autobiographical Novel] (inspirational source-creative work), [Historical Event] <-> [Documentary Film] (subject matter-artistic treatment). Focuses on creative origins rather than interpretive meanings."
    },
    {
        "category": "Symbolic-Representation",
        "definition": "Entity A is an artistic element that represents or symbolizes Entity B as a meaning or concept. Example: [Color Motif] <-> [Emotional State] (artistic symbol-represented meaning), [Literary Character] <-> [Societal Archetype] (creative element-symbolic meaning). Focuses on representational function rather than technical execution."
    },
    {
        "category": "Stylistic-Expression",
        "definition": "Entity A is a creator associated with Entity B as a distinctive artistic approach. Example: [Author] <-> [Narrative Voice] (creator-stylistic signature), [Filmmaker] <-> [Visual Aesthetic] (artist-characteristic style). Focuses on individual artistic expression rather than collective movements."
    },
    {
        "category": "Artistic-Context",
        "definition": "Entity A is an artistic movement situated within Entity B as a historical or cultural framework. Example: [Modernism] <-> [Post-World War Era] (artistic movement-historical period), [Magical Realism] <-> [Latin American Literature] (artistic style-cultural context). Focuses on contextual positioning rather than individual expression."
    },
    {
        "category": "Reception-Response",
        "definition": "Entity A is an artistic work that evokes Entity B as an audience reaction or critical evaluation. Example: [Controversial Performance] <-> [Public Debate] (artistic work-audience response), [Literary Classic] <-> [Scholarly Interpretation] (creative work-critical reception). Focuses on work-audience dynamics rather than creator intentions."
    },
    {
        "category": "Technical-Effect",
        "definition": "Entity A is an artistic technique that produces Entity B as an aesthetic outcome. Example: [Impressionist Brushwork] <-> [Light Perception] (painting technique-visual effect), [Unreliable Narrator] <-> [Narrative Suspense] (literary device-reader experience). Focuses on craft-effect relationships rather than thematic content."
    },
    {
        "category": "Cultural-Transformation",
        "definition": "Entity A is a cultural tradition or heritage that evolves into or influences Entity B as a contemporary expression. Example: [Oral Storytelling] <-> [Modern Fantasy Genre] (traditional form-contemporary development), [Folk Music] <-> [Popular Song Structure] (cultural heritage-modern adaptation). Focuses on artistic evolution rather than static preservation."
    },
    {
        "category": "Formal-Thematic",
        "definition": "Entity A represents artistic structure or form that shapes or contains Entity B as thematic content. Example: [Sonnet Form] <-> [Love Theme] (poetic structure-subject matter), [Three-Act Structure] <-> [Hero's Journey] (narrative form-thematic pattern). Focuses on form-content relationship rather than creative technique."
    },
    {
        "category": "Cross-Media",
        "definition": "Entity A is an artistic work or concept that transfers to or influences Entity B as a different artistic medium. Example: [Novel] <-> [Film Adaptation] (literary work-visual adaptation), [Musical Composition] <-> [Choreographic Interpretation] (sonic art-movement art). Focuses on inter-media transformation rather than within-medium influence."
    },
    {
        "category": "Patronage-Production",
        "definition": "Entity A provides support or sponsorship that enables Entity B as artistic creation. Example: [Arts Foundation] <-> [Experimental Theater] (funding source-artistic production), [Government Grant] <-> [Public Monument] (financial support-creative output). Focuses on enabling relationships rather than creative influences."
    }
]

# 政治与法律领域的特有实体关系
# Politics & Law domain-specific entity relations
POLITICS_LAW_RELATION_TYPES = [
    {
        "category": "Legislative-Impact",
        "definition": "Entity A is a law or policy that produces Entity B as a societal effect or outcome. Example: [Environmental Regulation] <-> [Emissions Reduction] (legal measure-policy outcome), [Tax Incentive] <-> [Economic Behavior] (legislative tool-societal response). Focuses on direct policy consequences rather than general governance."
    },
    {
        "category": "Rights-Responsibilities",
        "definition": "Entity A possesses legal entitlements that correspond to Entity B as obligations or duties. Example: [Citizen] <-> [Constitutional Rights] (rights-holder-legal entitlement), [Property Owner] <-> [Maintenance Obligations] (rights-bearer-corresponding duty). Focuses on reciprocal legal relationships rather than general social contracts."
    },
    {
        "category": "Jurisdictional-Authority",
        "definition": "Entity A is a governing body with legitimate power over Entity B as a domain or territory. Example: [Federal Agency] <-> [Regulated Industry] (authority-jurisdiction), [Municipal Government] <-> [City Limits] (governing entity-territorial scope). Focuses on formal authority boundaries rather than power dynamics."
    },
    {
        "category": "Political-Alignment",
        "definition": "Entity A is a political actor or position associated with Entity B as an ideological framework or value system. Example: [Political Party] <-> [Economic Philosophy] (political entity-ideological foundation), [Elected Official] <-> [Policy Platform] (political actor-value position). Focuses on political orientation rather than governance structures."
    },
    {
        "category": "Legal-Precedent",
        "definition": "Entity A is a judicial decision that establishes Entity B as a legal principle or interpretation. Example: [Supreme Court Case] <-> [Constitutional Interpretation] (court ruling-legal doctrine), [Judicial Opinion] <-> [Legal Standard] (legal decision-established principle). Focuses on case law development rather than statutory interpretation."
    },
    {
        "category": "Democratic-Representation",
        "definition": "Entity A is a political official or body that represents Entity B as a constituency or population. Example: [Elected Representative] <-> [District Residents] (public official-represented group), [Parliament] <-> [National Citizenry] (deliberative body-sovereign people). Focuses on representative democracy rather than direct governance."
    },
    {
        "category": "Constitutional-Structure",
        "definition": "Entity A is a foundational legal document that establishes Entity B as a governmental framework or principle. Example: [Constitution] <-> [Separation of Powers] (founding document-governance principle), [Charter] <-> [Fundamental Rights] (legal foundation-protected freedoms). Focuses on constitutional frameworks rather than specific legal provisions."
    },
    {
        "category": "Diplomatic-Relation",
        "definition": "Entity A is a state or international actor engaged with Entity B through formal international mechanisms. Example: [Nation] <-> [Treaty Obligation] (sovereign entity-international commitment), [International Organization] <-> [Member State] (multilateral body-participant country). Focuses on interstate relations rather than domestic politics."
    },
    {
        "category": "Power-Distribution",
        "definition": "Entity A represents political authority balanced against Entity B through institutional mechanisms. Example: [Executive Branch] <-> [Legislative Oversight] (governmental power-institutional check), [Federal Authority] <-> [State Sovereignty] (central power-local autonomy). Focuses on power balancing rather than policy implementation."
    },
    {
        "category": "Legal-Implementation",
        "definition": "Entity A is a legal standard or ruling enforced through Entity B as an implementation mechanism. Example: [Court Order] <-> [Enforcement Action] (legal decision-implementation measure), [Statute] <-> [Regulatory Framework] (legislation-enforcement structure). Focuses on law execution rather than law creation."
    }
]

# 为每个领域组合公共实体关系和特定领域实体关系
# Combine general and domain-specific entity relations for each domain
ACADEMIC_ENGINEERING_RELATIONS = GENERAL_ENTITY_RELATION_TYPES + ACADEMIC_ENGINEERING_RELATION_TYPES
FINANCE_BUSINESS_RELATIONS = GENERAL_ENTITY_RELATION_TYPES + FINANCE_BUSINESS_RELATION_TYPES
EDUCATION_RELATIONS = GENERAL_ENTITY_RELATION_TYPES + EDUCATION_RELATION_TYPES
LITERATURE_ART_RELATIONS = GENERAL_ENTITY_RELATION_TYPES + LITERATURE_ART_RELATION_TYPES
POLITICS_LAW_RELATIONS = GENERAL_ENTITY_RELATION_TYPES + POLITICS_LAW_RELATION_TYPES

# 域到实体关系的映射
# Domain to entity relations mapping
DOMAIN_TO_ENTITY_RELATIONS = {
    "academic_engineering": ACADEMIC_ENGINEERING_RELATIONS,
    "finance_business": FINANCE_BUSINESS_RELATIONS,
    "education": EDUCATION_RELATIONS,
    "literature_art": LITERATURE_ART_RELATIONS,
    "politics_law": POLITICS_LAW_RELATIONS
}


def get_entity_relations_by_domain(domain: str) -> List[Dict[str, str]]:

    if domain not in DOMAIN_TO_ENTITY_RELATIONS:
        valid_domains = list(DOMAIN_TO_ENTITY_RELATIONS.keys())
        raise ValueError(f"Invalid domain: {domain}. Valid domains are: {valid_domains}")
    
    return DOMAIN_TO_ENTITY_RELATIONS[domain]
