"""
traces.py
Ground-truth conversation traces parsed from C1-C10 for Recall@10 evaluation.
Each entry contains: trace_id, description, final_recommendations (ground truth).
"""

TRACES = [
    {
        "id": "C1",
        "description": "Senior leadership / CXO selection with OPQ and leadership reports",
        "final_recommendations": [
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P",
            },
            {
                "name": "OPQ Universal Competency Report 2.0",
                "url": "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
                "test_type": "P",
            },
            {
                "name": "OPQ Leadership Report",
                "url": "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
                "test_type": "P",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "We need a solution for senior leadership."},
            {"role": "assistant", "content": "Happy to help narrow that down. Who is this meant for?"},
            {"role": "user", "content": "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."},
            {"role": "assistant", "content": "For such roles, the OPQ32r is the right instrument. Is this for a newly created position, or developmental feedback?"},
            {"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."},
        ],
    },
    {
        "id": "C2",
        "description": "Senior Rust engineer — systems + networking, cognitive + personality",
        "final_recommendations": [
            {
                "name": "Smart Interview Live Coding",
                "url": "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
                "test_type": "K",
            },
            {
                "name": "Linux Programming (General)",
                "url": "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
                "test_type": "K",
            },
            {
                "name": "Networking and Implementation (New)",
                "url": "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
                "test_type": "K",
            },
            {
                "name": "SHL Verify Interactive G+",
                "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "test_type": "A",
            },
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"},
        ],
    },
    {
        "id": "C3",
        "description": "Entry-level contact centre agents, English US, two-stage design",
        "final_recommendations": [
            {
                "name": "SVAR Spoken English (US) (New)",
                "url": "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
                "test_type": "K",
            },
            {
                "name": "Contact Center Call Simulation (New)",
                "url": "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
                "test_type": "S",
            },
            {
                "name": "Entry Level Customer Serv - Retail & Contact Center",
                "url": "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
                "test_type": "P",
            },
            {
                "name": "Customer Service Phone Simulation",
                "url": "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
                "test_type": "B",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"},
        ],
    },
    {
        "id": "C4",
        "description": "Graduate financial analysts — numerical reasoning, finance knowledge, SJT, personality",
        "final_recommendations": [
            {
                "name": "SHL Verify Interactive – Numerical Reasoning",
                "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
                "test_type": "A",
            },
            {
                "name": "Financial Accounting (New)",
                "url": "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
                "test_type": "K",
            },
            {
                "name": "Basic Statistics (New)",
                "url": "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
                "test_type": "K",
            },
            {
                "name": "Graduate Scenarios",
                "url": "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
                "test_type": "B",
            },
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test."},
        ],
    },
    {
        "id": "C5",
        "description": "Sales org re-skilling audit — GSA, OPQ32r, MQ Sales Report, Sales Transformation",
        "final_recommendations": [
            {
                "name": "Global Skills Assessment",
                "url": "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
                "test_type": "K",
            },
            {
                "name": "Global Skills Development Report",
                "url": "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
                "test_type": "D",
            },
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P",
            },
            {
                "name": "OPQ MQ Sales Report",
                "url": "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
                "test_type": "P",
            },
            {
                "name": "Sales Transformation 2.0 - Individual Contributor",
                "url": "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
                "test_type": "P",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?"},
        ],
    },
    {
        "id": "C6",
        "description": "Plant operators, industrial safety-critical — DSI, Safety 8.0, Workplace Health & Safety",
        "final_recommendations": [
            {
                "name": "Manufac. & Indust. - Safety & Dependability 8.0",
                "url": "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
                "test_type": "P",
            },
            {
                "name": "Workplace Health and Safety (New)",
                "url": "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
                "test_type": "K",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?"},
        ],
    },
    {
        "id": "C7",
        "description": "Bilingual healthcare admin, South Texas, HIPAA + Spanish DSI/OPQ hybrid",
        "final_recommendations": [
            {
                "name": "HIPAA (Security)",
                "url": "https://www.shl.com/products/product-catalog/view/hipaa-security/",
                "test_type": "K",
            },
            {
                "name": "Medical Terminology (New)",
                "url": "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
                "test_type": "K",
            },
            {
                "name": "Microsoft Word 365 - Essentials (New)",
                "url": "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
                "test_type": "K",
            },
            {
                "name": "Dependability and Safety Instrument (DSI)",
                "url": "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
                "test_type": "P",
            },
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?"},
        ],
    },
    {
        "id": "C8",
        "description": "Admin assistants for Excel/Word — knowledge tests + simulations + OPQ32r",
        "final_recommendations": [
            {
                "name": "Microsoft Excel 365 (New)",
                "url": "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
                "test_type": "K",
            },
            {
                "name": "Microsoft Word 365 (New)",
                "url": "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
                "test_type": "K",
            },
            {
                "name": "MS Excel (New)",
                "url": "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
                "test_type": "K",
            },
            {
                "name": "MS Word (New)",
                "url": "https://www.shl.com/products/product-catalog/view/ms-word-new/",
                "test_type": "K",
            },
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "I need to quickly screen admin assistants for Excel and Word daily."},
        ],
    },
    {
        "id": "C9",
        "description": "Senior full-stack engineer (backend-leaning): Java, Spring, SQL, AWS, Docker, G+, OPQ32r",
        "final_recommendations": [
            {
                "name": "Core Java (Advanced Level) (New)",
                "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
                "test_type": "K",
            },
            {
                "name": "Spring (New)",
                "url": "https://www.shl.com/products/product-catalog/view/spring-new/",
                "test_type": "K",
            },
            {
                "name": "SQL (New)",
                "url": "https://www.shl.com/products/product-catalog/view/sql-new/",
                "test_type": "K",
            },
            {
                "name": "Amazon Web Services (AWS) Development (New)",
                "url": "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
                "test_type": "K",
            },
            {
                "name": "Docker (New)",
                "url": "https://www.shl.com/products/product-catalog/view/docker-new/",
                "test_type": "K",
            },
            {
                "name": "SHL Verify Interactive G+",
                "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "test_type": "A",
            },
            {
                "name": "Occupational Personality Questionnaire OPQ32r",
                "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "test_type": "P",
            },
        ],
        "seed_messages": [
            {
                "role": "user",
                "content": (
                    "Here's the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"
                    "\"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, "
                    "SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, "
                    "contribute to architectural decisions, and mentor mid-level engineers. Strong CI/CD and cloud-native "
                    "experience required.\""
                ),
            },
        ],
    },
    {
        "id": "C10",
        "description": "Graduate management trainees — Verify G+ and Graduate Scenarios (OPQ dropped by user)",
        "final_recommendations": [
            {
                "name": "SHL Verify Interactive G+",
                "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "test_type": "A",
            },
            {
                "name": "Graduate Scenarios",
                "url": "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
                "test_type": "B",
            },
        ],
        "seed_messages": [
            {"role": "user", "content": "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates."},
        ],
    },
]
