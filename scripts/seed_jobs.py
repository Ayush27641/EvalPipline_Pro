"""30 realistic tech job listings (fictional companies): Indian startups + global YC-style startups.

Imported by scripts/seed_data.py. Descriptions are normalised to single-line text at load time.
"""

from typing import Any, Dict, List, Optional


def _job(
    ref: str,
    title: str,
    company: str,
    location: str,
    employment_type: str,
    experience_level: str,
    salary_range: Optional[str],
    skills: List[str],
    description: str,
) -> Dict[str, Any]:
    return {
        "external_ref": ref,
        "title": title,
        "company": company,
        "location": location,
        "employment_type": employment_type,
        "experience_level": experience_level,
        "salary_range": salary_range,
        "skills": skills,
        "description": " ".join(description.split()),
    }


JOB_LISTINGS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------ India
    _job(
        "JOB-001", "Senior Backend Engineer (Python/FastAPI)", "Zestora Health",
        "Bengaluru, India (Hybrid)", "Full-time", "Senior (5-8 yrs)", "₹35-50 LPA + ESOPs",
        ["Python", "FastAPI", "PostgreSQL", "Redis", "AWS", "Docker", "Celery"],
        """Zestora Health builds the appointment, billing and clinical-records backbone for 1,200+
        clinics across India. You will own high-traffic FastAPI services handling 40M API calls a day,
        design PostgreSQL schemas for longitudinal patient records, and lead the move from Celery cron
        jobs to event-driven workflows on SQS. We expect deep async Python experience, a track record
        of running services in production on AWS, and comfort mentoring two or three mid-level
        engineers. Healthcare data handling (DPDP Act compliance) experience is a strong plus.
        Hybrid: three days a week in our Indiranagar office.""",
    ),
    _job(
        "JOB-002", "Machine Learning Engineer, LLM Platform", "Kaagaz AI",
        "Bengaluru, India (Hybrid)", "Full-time", "Mid-Senior (3-6 yrs)", "₹40-60 LPA",
        ["Python", "PyTorch", "LLM fine-tuning", "LoRA", "RAG", "vLLM", "Kubernetes", "Hugging Face"],
        """Kaagaz AI turns messy Indian legal and financial documents (scanned PDFs, Hindi/English mixed
        text) into structured data for banks and law firms. You will fine-tune open-weight LLMs (Llama,
        Qwen) with LoRA on domain data, build retrieval pipelines over millions of pages, and serve
        models with vLLM on GPU nodes in Kubernetes. Strong PyTorch fundamentals and hands-on experience
        evaluating RAG quality (groundedness, retrieval precision) are required. Experience with
        Indic-language NLP or OCR post-processing is a plus.""",
    ),
    _job(
        "JOB-003", "Platform Engineer (Go/Kubernetes)", "Vyomnet",
        "Hyderabad, India", "Full-time", "Mid-Senior (4-7 yrs)", "₹30-45 LPA",
        ["Go", "Kubernetes", "Terraform", "AWS", "Helm", "Prometheus", "gRPC"],
        """Vyomnet operates a multi-tenant Kubernetes platform used by 60+ product teams. You will write
        Go controllers and operators, own our Terraform modules for AWS (EKS, VPC, IAM), and build the
        internal developer portal that lets teams ship without filing tickets. We are looking for
        someone who has run Kubernetes in production at scale, understands networking (CNI, service
        mesh) deeply, and writes idiomatic, well-tested Go. On-call is shared across the platform team
        on a one-week-in-six rotation.""",
    ),
    _job(
        "JOB-004", "Data Engineer (Spark/Airflow)", "Dukaan Metrics",
        "Mumbai, India", "Full-time", "Mid (3-5 yrs)", "₹25-38 LPA",
        ["Apache Spark", "Airflow", "dbt", "Snowflake", "Python", "SQL", "Kafka"],
        """Dukaan Metrics provides retail analytics to 8,000 kirana-store distributors and FMCG brands.
        You will build and maintain Spark pipelines that process 2 TB of daily point-of-sale events,
        orchestrate them with Airflow, and model the warehouse in dbt on Snowflake. We need strong SQL,
        Python, and a bias for data quality: you will own freshness SLAs and anomaly alerting. Exposure
        to Kafka or streaming is helpful but not required. Based in our Lower Parel office with flexible
        hours.""",
    ),
    _job(
        "JOB-005", "Backend Developer (Node.js/TypeScript), Payments", "PaisaFlow",
        "Gurugram, India", "Full-time", "Mid (2-5 yrs)", "₹18-28 LPA",
        ["Node.js", "TypeScript", "PostgreSQL", "Kafka", "UPI", "REST APIs", "Redis"],
        """PaisaFlow is a UPI-first payments gateway for small merchants. You will build settlement and
        reconciliation services in TypeScript on Node.js, integrate with NPCI and bank partner APIs, and
        work on idempotent, exactly-once payment flows backed by PostgreSQL and Kafka. Candidates should
        understand transactional consistency, retries and ledgers, and care about audit trails. Prior
        fintech or payments experience is strongly preferred. Five days in office in Cyber City.""",
    ),
    _job(
        "JOB-006", "Junior Software Engineer, Backend (Python)", "Nimbus Kart",
        "Pune, India", "Full-time", "Entry level (0-2 yrs)", "₹8-14 LPA",
        ["Python", "Django", "REST APIs", "PostgreSQL", "Git", "Docker basics"],
        """Nimbus Kart is a quick-commerce startup delivering groceries in 20 minutes across Pune and
        Nashik. As a junior backend engineer you will build Django REST endpoints for the order and
        inventory services, write unit tests, fix production bugs with guidance from senior engineers,
        and gradually take ownership of a module. We hire for fundamentals: clean Python, SQL basics,
        Git hygiene and curiosity. Fresh graduates with strong personal or open-source projects are
        welcome. Structured 12-week onboarding with a dedicated mentor.""",
    ),
    _job(
        "JOB-007", "Site Reliability Engineer", "Chalo Logistics",
        "Bengaluru, India (Hybrid)", "Full-time", "Mid-Senior (3-6 yrs)", "₹28-40 LPA",
        ["Linux", "Kubernetes", "Prometheus", "Grafana", "Python", "Terraform", "Incident management"],
        """Chalo Logistics runs real-time tracking for 50,000 intercity trucks. You will own availability
        of the fleet-tracking platform: define SLOs, build Prometheus/Grafana observability, automate
        capacity management on Kubernetes, and lead blameless post-mortems. Strong Linux troubleshooting,
        scripting in Python, and experience with infrastructure-as-code are required. You will be part
        of a 24x7 on-call rotation with compensated shifts. Office in Koramangala, with work-from-home
        two days a week.""",
    ),
    _job(
        "JOB-008", "Applied AI Engineer, Conversational Agents", "Bolo Labs",
        "Remote (India)", "Full-time", "Mid-Senior (3-6 yrs)", None,
        ["Python", "LLMs", "RAG", "Speech-to-text", "Prompt engineering", "FastAPI", "LLM evaluation"],
        """Bolo Labs builds Hindi and English voice agents for customer support in insurance and lending.
        You will design multi-turn conversational flows on top of LLMs, build retrieval over policy
        documents, and set up automated evaluation harnesses (LLM-as-a-judge, regression suites) so
        prompt changes never silently degrade quality. Experience shipping LLM features to production,
        handling code-switched Hindi/English input, and working with ASR/TTS vendors is valued. Fully
        remote within India; quarterly team offsites.""",
    ),
    _job(
        "JOB-009", "Full-Stack Engineer (React + Django)", "Pathshala Pro",
        "Hyderabad, India (Hybrid)", "Full-time", "Mid (2-5 yrs)", "₹20-32 LPA",
        ["React", "TypeScript", "Django", "PostgreSQL", "GraphQL", "AWS"],
        """Pathshala Pro is an ed-tech platform used by 900 schools for assessments and parent
        communication. You will build features end to end: React/TypeScript frontends, Django backends,
        and the GraphQL layer between them. Expect to work closely with product and design, ship weekly,
        and own quality through tests and monitoring. Experience with accessibility, low-bandwidth
        optimisation, or multilingual UIs is a plus. Based in Gachibowli; hybrid schedule.""",
    ),
    _job(
        "JOB-010", "Android Engineer (Kotlin)", "Rupay Credit",
        "Mumbai, India", "Full-time", "Mid-Senior (3-6 yrs)", "₹22-35 LPA",
        ["Kotlin", "Jetpack Compose", "Android SDK", "MVVM", "Retrofit", "Mobile security"],
        """Rupay Credit offers credit lines to gig workers through an Android-first app with 4M installs.
        You will build and maintain the app in Kotlin with Jetpack Compose, implement secure KYC and
        device-binding flows, and optimise performance for low-end devices and patchy networks. Solid
        understanding of the Android lifecycle, offline-first architecture, and app security (root
        detection, certificate pinning) is expected. Fintech experience preferred. Office in Andheri
        East.""",
    ),
    _job(
        "JOB-011", "Security Engineer, Application Security", "SecureKart",
        "Bengaluru, India (Hybrid)", "Full-time", "Senior (5-8 yrs)", "₹32-50 LPA",
        ["Threat modelling", "SAST/DAST", "OWASP", "Python", "AWS security", "Secure code review"],
        """SecureKart is an e-commerce marketplace handling 3M orders a month. You will run the
        application security program: threat-model new features, perform secure code reviews across
        Java and Python services, operate SAST/DAST tooling in CI, and triage bug-bounty reports. We
        want someone who can write Python automation, explain vulnerabilities to developers without
        condescension, and prioritise by real risk. CISSP/OSCP are nice but not required. Hybrid in
        Whitefield.""",
    ),
    _job(
        "JOB-012", "Senior Rust Engineer, Storage Systems", "Dhruva Systems",
        "Bengaluru, India", "Full-time", "Senior (6-10 yrs)", "₹45-70 LPA + ESOPs",
        ["Rust", "Distributed systems", "Raft", "Linux internals", "io_uring", "Performance engineering"],
        """Dhruva Systems is building a distributed object store optimised for AI training workloads,
        written in Rust. You will work on the replication layer (Raft-based consensus), the on-disk
        format, and the io_uring-based I/O path, with a relentless focus on tail latency and throughput.
        We expect production Rust experience, a deep understanding of Linux storage and networking
        internals, and comfort reasoning about correctness under failure. Experience with Ceph, MinIO,
        or similar systems is a plus. Office in HSR Layout; relocation assistance available.""",
    ),
    _job(
        "JOB-013", "DevOps Engineer (Azure)", "Meridian Works",
        "Noida, India", "Full-time", "Mid (3-5 yrs)", "₹15-24 LPA",
        ["Azure", "Terraform", "Azure DevOps", "Docker", "AKS", "PowerShell", "Bash"],
        """Meridian Works provides HR and payroll SaaS to mid-size enterprises. You will manage our Azure
        estate with Terraform, maintain Azure DevOps CI/CD pipelines for .NET and Node services, run
        AKS clusters, and drive cost optimisation. Candidates should be comfortable with scripting
        (PowerShell/Bash), Azure networking and identity (Entra ID), and on-call for infrastructure
        incidents. Azure certifications (AZ-104/AZ-400) are a plus. Five days a week in our Sector 62
        office.""",
    ),
    _job(
        "JOB-014", "Data Scientist, Growth Analytics", "Swasthya Insights",
        "Chennai, India (Hybrid)", "Full-time", "Mid (2-5 yrs)", "₹20-30 LPA",
        ["Python", "SQL", "Experimentation", "Causal inference", "Pandas", "Looker"],
        """Swasthya Insights helps diagnostic labs grow through a B2B2C app. You will design and analyse
        A/B tests, build retention and LTV models, and partner with product managers to turn findings
        into roadmap decisions. Strong SQL and Python (pandas, statsmodels), rigorous experimentation
        methodology, and clear communication are essential. Experience with causal inference methods
        or marketing mix modelling is a plus. Hybrid in Guindy.""",
    ),
    _job(
        "JOB-015", "Software Engineering Intern, Backend (Python/Go)", "Vyomnet",
        "Hyderabad, India", "Internship (6 months)", "Student / Fresher",
        "₹40,000-60,000 per month stipend",
        ["Python", "Go", "REST APIs", "Git", "Linux", "Docker basics"],
        """Join Vyomnet's platform team for a six-month internship. You will build internal tooling in
        Python and Go, write integration tests for our Kubernetes operators, and ship at least one
        feature to production under a mentor's guidance. We look for final-year students or recent
        graduates with solid programming fundamentals, familiarity with Linux and Git, and an interest
        in infrastructure. High-performing interns receive a pre-placement offer for the Junior Platform
        Engineer role. On-site in Hitec City.""",
    ),
    # ------------------------------------------------------------------ Global (YC-style startups)
    _job(
        "JOB-016", "Founding Backend Engineer", "Lumen Ledger (YC W24)",
        "San Francisco, CA, USA (Hybrid)", "Full-time", "Senior (5+ yrs)",
        "$160,000-$210,000 + 0.5-1.5% equity",
        ["Python", "FastAPI", "PostgreSQL", "Stripe API", "AWS", "Event sourcing"],
        """Lumen Ledger (YC W24) is building real-time accounting infrastructure for marketplaces: every
        payout, refund and fee becomes a double-entry journal automatically. As the first backend hire
        you will design the ledger core in Python/FastAPI on PostgreSQL, integrate Stripe Connect and
        Adyen, and set engineering culture alongside the two technical founders. We want someone who
        has shipped financial or high-correctness systems, enjoys ambiguity, and writes code that is
        boring in the best way. Hybrid in SoMa, three days a week.""",
    ),
    _job(
        "JOB-017", "Senior Machine Learning Engineer, Recommendations", "Fable Commerce",
        "New York, NY, USA (Hybrid)", "Full-time", "Senior (5-8 yrs)", "$180,000-$240,000 + equity",
        ["PyTorch", "Recommender systems", "Learning-to-rank", "Feature stores", "Spark", "A/B testing"],
        """Fable Commerce powers personalised storefronts for 2,000 independent fashion brands. You will
        own the ranking and recommendation models that drive 35% of GMV: two-tower retrieval,
        learning-to-rank, and real-time feature pipelines. Strong PyTorch skills, experience taking
        recommender models from offline metrics to online A/B wins, and comfort with large-scale data in
        Spark are required. Experience with LLM-based product understanding is a plus. Based in Flatiron
        with a hybrid schedule.""",
    ),
    _job(
        "JOB-018", "Staff Software Engineer, Distributed Systems", "Quillstream",
        "Remote (USA)", "Full-time", "Staff (8+ yrs)", "$220,000-$280,000 + equity",
        ["Go", "Kafka", "Distributed consensus", "Multi-region replication", "PostgreSQL", "Kubernetes"],
        """Quillstream is a streaming data platform (think managed Kafka plus stream processing) used by
        400 companies. You will lead the design of our multi-region replication layer in Go, drive
        consistency and failover semantics, and mentor senior engineers across three teams. We expect
        deep distributed-systems expertise (consensus, replication, failure modes), production Go, and
        a history of technical leadership without direct reports. Fully remote within the US with
        quarterly in-person weeks.""",
    ),
    _job(
        "JOB-019", "AI Infrastructure Engineer, GPU Clusters", "Tensorhaus",
        "Berlin, Germany", "Full-time", "Mid-Senior (3-7 yrs)", "€95,000-€130,000",
        ["Kubernetes", "CUDA", "Slurm", "NVIDIA Triton", "vLLM", "Python", "Linux", "InfiniBand"],
        """Tensorhaus runs training and inference clusters for European AI labs. You will operate 1,000+
        GPU Kubernetes and Slurm clusters, build inference serving with Triton and vLLM, optimise
        InfiniBand networking and storage throughput, and write the Python tooling that keeps
        utilisation above 85%. Strong Linux, Kubernetes and CUDA ecosystem knowledge are required;
        experience with NCCL debugging is a big plus. Based in Berlin-Kreuzberg; relocation support
        offered for candidates moving to Germany.""",
    ),
    _job(
        "JOB-020", "LLM Evaluation Engineer", "Veridict AI (YC S24)",
        "Remote (Global)", "Full-time", "Mid-Senior (3-6 yrs)", "$140,000-$190,000 + equity",
        ["Python", "LLM-as-a-judge", "Evaluation design", "Statistics", "FastAPI", "PostgreSQL", "CI/CD"],
        """Veridict AI (YC S24) builds evaluation infrastructure for teams shipping LLM products. You will
        design judge prompts and calibration datasets, build async evaluation pipelines that score
        faithfulness, relevancy and retrieval precision at scale, and ship the regression gates
        customers run in CI. Candidates need strong Python, a statistical mindset (inter-rater
        agreement, confidence intervals), and hands-on experience with RAG systems. Fully remote,
        async-first team spread from Bengaluru to San Francisco.""",
    ),
    _job(
        "JOB-021", "Backend Engineer (Elixir)", "Pulse Health",
        "London, UK (Hybrid)", "Full-time", "Mid-Senior (3-6 yrs)", "£85,000-£110,000",
        ["Elixir", "Phoenix", "PostgreSQL", "FHIR", "OTP", "GraphQL"],
        """Pulse Health connects NHS trusts with remote-monitoring devices for chronic-care patients. You
        will build Elixir/Phoenix services that ingest device telemetry, map it to FHIR resources, and
        expose GraphQL APIs to clinician dashboards. Experience with OTP, PostgreSQL and healthcare
        interoperability is valued; we will teach FHIR to strong Elixir engineers. Hybrid in Shoreditch,
        two days a week.""",
    ),
    _job(
        "JOB-022", "Senior Data Platform Engineer", "Harbor Analytics",
        "Singapore", "Full-time", "Senior (5-8 yrs)", "S$140,000-S$180,000",
        ["Apache Spark", "Apache Flink", "Apache Iceberg", "Kafka", "AWS", "Python", "Scala"],
        """Harbor Analytics provides supply-chain visibility for shipping lines across Southeast Asia. You
        will build our lakehouse on Iceberg with Spark and Flink, design streaming ingestion from Kafka,
        and own the platform that data scientists and analysts build on. We expect strong experience
        with Spark internals, streaming semantics (exactly-once, watermarks), and AWS data services.
        Scala or Python are both fine. Based in Singapore; employment pass sponsorship available.""",
    ),
    _job(
        "JOB-023", "MLOps Engineer, Model Serving", "Northwind Robotics",
        "Toronto, Canada (Hybrid)", "Full-time", "Mid (3-5 yrs)", "C$130,000-C$170,000",
        ["Kubernetes", "MLflow", "NVIDIA Triton", "GitHub Actions", "Python", "Terraform", "Monitoring"],
        """Northwind Robotics builds perception systems for warehouse robots. You will own the path from
        trained model to robot fleet: model registry (MLflow), CI/CD with GitHub Actions, canary
        deployments of vision models via Triton, and drift monitoring in production. Strong Kubernetes
        and Python skills and experience with GPU inference optimisation are required. Robotics or edge
        deployment experience is a plus. Hybrid in downtown Toronto.""",
    ),
    _job(
        "JOB-024", "Senior Full-Stack Engineer (TypeScript/Next.js)", "Canopy Legal (YC W23)",
        "Remote (USA/Canada)", "Full-time", "Senior (5+ yrs)", "$170,000-$200,000 + equity",
        ["TypeScript", "Next.js", "React", "Node.js", "PostgreSQL", "Prisma", "LLM APIs"],
        """Canopy Legal (YC W23) automates contract review for in-house legal teams. You will build the
        product end to end in TypeScript: Next.js frontends, Node.js services, PostgreSQL via Prisma,
        and LLM-powered features for clause extraction. We want product-minded engineers who ship fast,
        write tests, and sweat UX details. Experience integrating LLM APIs into production features is
        a plus. Fully remote across US and Canadian time zones.""",
    ),
    _job(
        "JOB-025", "Backend Engineer (Java/Spring), Marketplace", "Orbital Freight",
        "Amsterdam, Netherlands (Hybrid)", "Full-time", "Mid (3-5 yrs)", "€80,000-€105,000",
        ["Java 17", "Spring Boot", "Kafka", "PostgreSQL", "Microservices", "Docker"],
        """Orbital Freight is a digital freight marketplace matching shippers with carriers across
        Europe. You will build Spring Boot microservices for pricing, booking and tracking, publish
        domain events to Kafka, and improve resilience across a growing service landscape. Solid Java
        17, Spring and SQL skills plus an understanding of distributed-systems trade-offs are required.
        Based in Amsterdam Zuidas; 30% remote and a relocation budget for EU hires.""",
    ),
    _job(
        "JOB-026", "Research Engineer, Speech & Audio ML", "Sonara Labs",
        "Remote (Europe, CET ±2)", "Full-time", "Mid-Senior (3-7 yrs)", "€110,000-€150,000",
        ["PyTorch", "ASR", "TTS", "Diffusion models", "Audio signal processing", "Python", "CUDA"],
        """Sonara Labs builds low-latency speech models for real-time translation. You will train and
        optimise ASR and TTS models in PyTorch, experiment with diffusion and flow-matching approaches
        for voice synthesis, and bring research to production with streaming inference under 300 ms.
        We expect strong ML fundamentals, experience with audio data pipelines, and a publication or
        open-source track record. Fully remote within European time zones (CET ±2).""",
    ),
    _job(
        "JOB-027", "Security Engineer, Cloud Infrastructure", "Keelhaul Security",
        "Austin, TX, USA (Hybrid)", "Full-time", "Senior (5-8 yrs)", "$165,000-$205,000",
        ["AWS IAM", "GCP", "Terraform", "Detection engineering", "Python", "Kubernetes security", "SIEM"],
        """Keelhaul Security is a managed detection and response provider for cloud-native companies.
        You will harden our own multi-cloud infrastructure, build detection rules for AWS and GCP
        control-plane abuse, and write Python tooling to automate IAM least-privilege reviews. Deep
        knowledge of AWS IAM, Kubernetes security, and Terraform is required. Experience with SIEM
        pipelines or incident response is a plus. Hybrid in Austin.""",
    ),
    _job(
        "JOB-028", "Compiler Engineer, ML Compilers", "Halide Compute",
        "Seattle, WA, USA", "Full-time", "Senior (5+ yrs)", "$190,000-$250,000 + equity",
        ["C++", "MLIR", "LLVM", "GPU kernels", "CUDA", "Python", "Performance optimisation"],
        """Halide Compute builds a compiler that maps PyTorch graphs to custom AI accelerators and GPUs.
        You will work on MLIR dialects, operator fusion, memory planning and code generation for GPU
        kernels. Strong modern C++, compiler internals (LLVM/MLIR), and experience with performance
        work on GPUs are required. Low-level systems programmers from adjacent domains (Rust, HPC,
        graphics) are encouraged to apply. On-site in South Lake Union with relocation support.""",
    ),
    _job(
        "JOB-029", "Developer Advocate, AI Developer Tools", "Promptly Dev",
        "Remote (USA)", "Full-time", "Mid (3-5 yrs)", "$150,000-$180,000",
        ["Python", "Technical writing", "Public speaking", "LLM APIs", "Demos", "Open source"],
        """Promptly Dev makes an open-source SDK for building and testing LLM applications. You will
        build demo apps, write tutorials, speak at meetups and conferences, and feed developer pain
        points back to the product team. We want an engineer who codes daily in Python, explains complex
        ideas simply, and has shipped at least one LLM-powered project. Fully remote within the US with
        a travel budget.""",
    ),
    _job(
        "JOB-030", "Engineering Manager, Payments Platform", "Mercurio Pay",
        "Dublin, Ireland (Hybrid)", "Full-time", "Manager (8+ yrs, 2+ managing)",
        "€130,000-€160,000 + equity",
        ["Engineering leadership", "Go", "Python", "PCI DSS", "Kafka", "Hiring", "Roadmap planning"],
        """Mercurio Pay processes card and open-banking payments for European SaaS companies. You will
        lead a team of eight backend engineers (Go and Python) that owns the payments core, drive PCI
        DSS compliance work, and shape the roadmap with product and risk. We are looking for a manager
        who still reads code, has run on-call teams, and has grown engineers into senior roles. Hybrid
        in Dublin Docklands.""",
    ),
]
