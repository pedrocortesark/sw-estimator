"""Golden dataset for the sw-estimator evaluation suite.

Each Golden case pairs a realistic project transcript with metadata that
defines the *success criteria* for that case — not exact outputs, but the
observable properties a correct estimation must satisfy.

Metadata fields
---------------
category : str
    One of: small_project | medium_project | large_project |
            ambiguous | contradictory | multilingual
expected_hours_range : tuple[int, int]
    Inclusive range [low, high] that an expert human would consider
    reasonable. Used by range-based assertions in tests.
expected_components : list[str]
    Phase or component keywords that *must* appear somewhere in the
    estimation output (case-insensitive substring match is fine).
key_risks : list[str]
    Risk topics that a high-quality estimation should flag.  These are
    used in LLM-graded metrics, not in exact-match assertions.
notes : str
    Human-readable rationale explaining what this case is testing and
    why the expected range was chosen.
"""

from deepeval.dataset import EvaluationDataset, Golden

# ---------------------------------------------------------------------------
# Individual Golden cases
# ---------------------------------------------------------------------------

_GOLDENS: list[Golden] = [
    # ------------------------------------------------------------------ #
    # 1. Small project — landing page with contact form                   #
    # ------------------------------------------------------------------ #
    Golden(
        input=(
            "The client needs a corporate landing page for their new SaaS product. "
            "It must have five sections: hero, features, pricing table with three tiers, "
            "a contact form that sends emails, and a footer. "
            "Design is provided in Figma. No authentication, no database — just a "
            "static site with a serverless function for the form submission."
        ),
        expected_output=None,  # No exact answer; criteria live in metadata
        additional_metadata={
            "category": "small_project",
            "expected_hours_range": (16, 40),
            "expected_components": ["frontend", "form"],
            "key_risks": [
                "Figma handoff quality",
                "email deliverability",
                "cross-browser compatibility",
            ],
            "notes": (
                "Canonical small project: bounded scope, no backend, design already "
                "exists. Any estimate below 16 h ignores QA and deployment; above "
                "40 h suggests scope inflation."
            ),
        },
    ),
    # ------------------------------------------------------------------ #
    # 2. Medium project — admin dashboard                                 #
    # ------------------------------------------------------------------ #
    Golden(
        input=(
            "We need an internal admin dashboard for our operations team. "
            "It needs user management with role-based access control (admin, editor, "
            "viewer), an audit log of all user actions, a data table showing orders "
            "with filtering and CSV export, and automated weekly email reports "
            "summarising KPIs. The backend should be a REST API backed by PostgreSQL. "
            "We already have the authentication service; we just need to integrate it."
        ),
        expected_output=None,
        additional_metadata={
            "category": "medium_project",
            "expected_hours_range": (200, 400),
            "expected_components": ["backend", "frontend", "auth", "email", "database"],
            "key_risks": [
                "RBAC complexity",
                "audit log volume and retention",
                "email scheduling reliability",
                "existing auth integration contract",
            ],
            "notes": (
                "Mid-size internal tool. Auth is pre-existing but integration "
                "still carries risk. CSV export and scheduled emails are "
                "often underestimated."
            ),
        },
    ),
    # ------------------------------------------------------------------ #
    # 3. Large project — multi-payment e-commerce with async processing   #
    # ------------------------------------------------------------------ #
    Golden(
        input=(
            "Build a B2C e-commerce platform for a fashion retailer. "
            "The platform must integrate with three payment gateways: Stripe for cards, "
            "PayPal for wallets, and a local PSP for bank transfers. "
            "Orders trigger an asynchronous fulfilment pipeline: inventory reservation, "
            "warehouse pick-list generation via an external WMS API, "
            "and real-time shipment tracking via two courier APIs. "
            "The catalogue has 50 000 SKUs with variant support. "
            "We need a React storefront, a backoffice for catalogue management, "
            "and a mobile app (iOS + Android) using React Native. "
            "PCI-DSS compliance is mandatory."
        ),
        expected_output=None,
        additional_metadata={
            "category": "large_project",
            "expected_hours_range": (2000, 6000),
            "expected_components": [
                "frontend",
                "mobile",
                "backend",
                "payment",
                "async",
                "integration",
                "database",
            ],
            "key_risks": [
                "PCI-DSS compliance scope",
                "three-PSP reconciliation",
                "WMS and courier API stability",
                "50k-SKU catalogue performance",
                "React Native maintenance overhead",
            ],
            "notes": (
                "Complex project with three external payment integrations, "
                "async pipeline, and mobile. PCI-DSS alone adds significant "
                "security-review and penetration-testing effort that junior "
                "estimators routinely forget."
            ),
        },
    ),
    # ------------------------------------------------------------------ #
    # 4. Ambiguous case — missing critical details                        #
    # ------------------------------------------------------------------ #
    Golden(
        input=(
            "We want to build a marketplace. Users should be able to list things "
            "and buy things. We also need notifications and a dashboard. "
            "It should look modern and work on mobile too."
        ),
        expected_output=None,
        additional_metadata={
            "category": "ambiguous",
            "expected_hours_range": (0, 9999),  # Any range is acceptable
            "expected_components": [],  # Cannot assert components without detail
            "key_risks": [
                "undefined payment flow",
                "missing trust & safety requirements",
                "no performance or scale targets",
                "notification channel unspecified",
            ],
            "notes": (
                "Deliberately vague. The estimation should surface the missing "
                "information rather than invent scope. A good output explicitly "
                "lists assumptions and flags that the range is wide due to "
                "unspecified requirements."
            ),
        },
    ),
    # ------------------------------------------------------------------ #
    # 5. Contradictory case — internal inconsistencies                    #
    # ------------------------------------------------------------------ #
    Golden(
        input=(
            "The client confirmed the project must be delivered in two weeks. "
            "The scope includes a full ERP system with inventory, HR, payroll, "
            "CRM, and a custom reporting engine. Budget is fixed at $5 000. "
            "The tech stack must use microservices with event sourcing. "
            "The team will consist of one junior developer working part-time."
        ),
        expected_output=None,
        additional_metadata={
            "category": "contradictory",
            "expected_hours_range": (0, 9999),
            "expected_components": [],
            "key_risks": [
                "timeline physically impossible",
                "budget vs scope mismatch",
                "single junior developer for ERP",
                "microservices complexity with part-time team",
            ],
            "notes": (
                "The constraints are mutually exclusive: ERP + microservices + "
                "event sourcing cannot be delivered in two weeks by one part-time "
                "junior dev for $5 000. The estimation must call out the "
                "contradictions explicitly rather than produce a fictional plan."
            ),
        },
    ),
    # ------------------------------------------------------------------ #
    # 6. Data-intensive project — ML pipeline                            #
    # ------------------------------------------------------------------ #
    Golden(
        input=(
            "We need a churn-prediction pipeline for our SaaS product. "
            "The pipeline should ingest daily usage events from Kafka, "
            "run feature engineering in Spark, train an XGBoost model weekly, "
            "serve predictions via a REST endpoint with sub-200 ms p99 latency, "
            "and expose a monitoring dashboard with model drift alerts. "
            "Training data covers 18 months of history (~2 TB). "
            "The model must be explainable: feature importance per prediction."
        ),
        expected_output=None,
        additional_metadata={
            "category": "medium_project",
            "expected_hours_range": (400, 900),
            "expected_components": [
                "data_ingestion",
                "feature_engineering",
                "model_training",
                "serving",
                "monitoring",
            ],
            "key_risks": [
                "Kafka consumer reliability",
                "Spark cluster sizing for 2 TB",
                "model retraining cadence governance",
                "explainability library integration",
                "p99 latency SLA under load",
            ],
            "notes": (
                "ML engineering projects are frequently underestimated because "
                "feature engineering and monitoring are treated as afterthoughts. "
                "The latency SLA and explainability requirement add non-trivial "
                "complexity to the serving layer."
            ),
        },
    ),
    # ------------------------------------------------------------------ #
    # 7. Multilingual case — Spanish transcript                          #
    # ------------------------------------------------------------------ #
    Golden(
        input=(
            "Necesitamos una aplicación móvil para gestión de turnos en clínicas. "
            "El médico debe poder ver su agenda diaria, confirmar o cancelar citas, "
            "y recibir notificaciones push cuando se agenda una nueva cita. "
            "El paciente puede reservar, reprogramar o cancelar su cita. "
            "Necesitamos integración con el sistema de historia clínica existente "
            "mediante una API REST que ya está documentada. "
            "La app debe funcionar en iOS y Android."
        ),
        expected_output=None,
        additional_metadata={
            "category": "multilingual",
            "expected_hours_range": (250, 500),
            "expected_components": [
                "mobile",
                "backend",
                "notifications",
                "integration",
            ],
            "key_risks": [
                "legacy EHR API stability",
                "push notification reliability",
                "timezone handling for appointment scheduling",
                "HIPAA / patient data privacy compliance",
            ],
            "notes": (
                "Spanish-language transcript. Tests that the system handles "
                "non-English input without degrading estimation quality. "
                "The existing EHR integration is a common risk in healthcare "
                "projects that estimators underweight."
            ),
        },
    ),
    # ------------------------------------------------------------------ #
    # 8. Infrastructure / DevOps — platform migration                    #
    # ------------------------------------------------------------------ #
    Golden(
        input=(
            "We want to migrate our monolithic Rails application (120k lines of code, "
            "running on bare-metal servers) to Kubernetes on AWS EKS. "
            "The migration should be zero-downtime. We have 14 background job types "
            "using Sidekiq, a PostgreSQL database (800 GB), and an Elasticsearch "
            "cluster for search. CI/CD currently runs on Jenkins; "
            "we want to move to GitHub Actions at the same time. "
            "After the migration, response time at p95 must not exceed current values."
        ),
        expected_output=None,
        additional_metadata={
            "category": "large_project",
            "expected_hours_range": (600, 1800),
            "expected_components": [
                "infrastructure",
                "ci_cd",
                "database_migration",
                "containerisation",
                "monitoring",
            ],
            "key_risks": [
                "zero-downtime cutover with 800 GB Postgres",
                "Sidekiq job compatibility in containers",
                "Elasticsearch index migration",
                "Jenkins-to-GHA feature parity",
                "p95 performance regression after containerisation",
            ],
            "notes": (
                "Infrastructure migrations are notoriously underestimated. "
                "Combining a Kubernetes migration with a CI/CD platform switch "
                "in a single project multiplies coordination risk. "
                "The performance SLA requires load testing which adds scope."
            ),
        },
    ),
]

# ---------------------------------------------------------------------------
# Public dataset
# ---------------------------------------------------------------------------

golden_dataset = EvaluationDataset(goldens=_GOLDENS)
