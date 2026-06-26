"""Golden set de 5 consultas representativas del dominio.

Cada consulta es una descripción realista de un proyecto a estimar, como la que
podría dar un cliente en una reunión. Para cada consulta se anotan manualmente
los presupuestos del dataset que son realmente relevantes.

Los presupuestos del dataset están en inglés, así que las consultas también.
"""

GOLDEN_SET = [
    {
        "query_id": "Q1",
        "description": "Mobile banking app with OAuth authentication and regulatory compliance",
        "query": "We need a mobile banking application with secure OAuth 2.0 authentication, "
                 "PSD2 compliance for open banking, and a transaction ledger with audit trails. "
                 "The app should support real-time notifications for transactions and security events.",
        "relevant_budgets": ["BUD-2024-001", "BUD-2024-002", "BUD-2024-003"],
        "rationale": "Finance sector projects with OAuth, PSD2, and ledger components"
    },
    {
        "query_id": "Q2",
        "description": "E-commerce platform with product catalog and checkout",
        "query": "Building an online store with a product catalog, shopping cart, and checkout flow. "
                 "Need payment integration, order management, and email notifications for receipts. "
                 "Should support multiple currencies and promotional discounts.",
        "relevant_budgets": ["BUD-2024-005", "BUD-2024-006", "BUD-2024-007", "BUD-2024-017"],
        "rationale": "E-commerce projects with catalog, cart, checkout components"
    },
    {
        "query_id": "Q3",
        "description": "Telemedicine platform with video consultations",
        "query": "Healthcare platform for patient appointments and telemedicine. Need video consultation "
                 "capabilities, integration with hospital EHR systems using HL7 FHIR standards, "
                 "and GDPR-compliant consent management for protected health information.",
        "relevant_budgets": ["BUD-2024-009", "BUD-2024-010", "BUD-2024-011", "BUD-2024-012"],
        "rationale": "Healthcare projects with telemedicine, FHIR, and compliance components"
    },
    {
        "query_id": "Q4",
        "description": "Industrial IoT system for predictive maintenance",
        "query": "Factory monitoring system that collects sensor data from machines via MQTT, "
                 "detects anomalies using predictive models, and integrates with existing SCADA dashboards. "
                 "Need real-time telemetry ingestion and maintenance scheduling based on equipment health.",
        "relevant_budgets": ["BUD-2024-013", "BUD-2024-014", "BUD-2024-015"],
        "rationale": "Industrial projects with IoT, telemetry, and predictive maintenance"
    },
    {
        "query_id": "Q5",
        "description": "Real-time payment gateway with fraud detection",
        "query": "Payment processing system for merchants with real-time transaction processing, "
                 "fraud detection using machine learning, and a dashboard for tracking payments and refunds. "
                 "Need reconciliation workflows and compliance with PCI DSS standards.",
        "relevant_budgets": ["BUD-2024-003", "BUD-2024-004", "BUD-2024-016"],
        "rationale": "Finance projects with payments, fraud detection, and reconciliation"
    }
]
