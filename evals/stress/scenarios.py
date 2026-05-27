"""Synthetic multi-turn stress scenarios for the sw-estimator evaluation suite.

Three conversation profiles, each designed to exercise a different failure mode
of the session memory system.  Each profile declares a *fact-tracker*: a flat
list of :class:`FactAssertion` objects that :class:`MemoryDriftMetric` (Block 4)
evaluates after every subsequent turn to detect silent memory drift.

Profiles
--------
GROWING_PROJECT
    Coherent requirements accumulate turn by turn (auth → multi-tenancy →
    audit log → CSV export → …).  Verifies the cost curve and that the
    project name set on turn 1 survives unchanged to turn 20.

PIVOTING_PROJECT
    Turn 5 abandons React + Node.js and replaces the entire stack with
    Flutter + FastAPI.  Verifies that the pivot is detected (Flutter is
    added) while also surfacing the expected "stale technology" drift
    (React stays in ``mentioned_technologies`` because the accumulator
    never removes items).

CONTRADICTING_PROJECT
    Turn 3 states a 30 k€ budget ceiling; turn 8 raises it to 80 k€.
    Verifies which constraint ends up in ``agreed_scope`` / ``anchors`` /
    ``accumulated_summary`` and whether the tier resolver reflects the
    revised budget correctly.

N_VALUES
--------
The canonical set of turn-counts to evaluate each profile against::

    N_VALUES = (1, 3, 6, 10, 20)

Usage
-----
::

    from evals.stress.scenarios import ALL_PROFILES, N_VALUES

    for profile in ALL_PROFILES:
        for n in N_VALUES:
            turns  = profile.script_for(n)
            # … run each TurnScript.transcript against a live session …
            facts  = profile.all_facts_up_to(n)
            for fact in facts:
                passed = fact.check(session)

MemoryDriftMetric contract
--------------------------
A fact is considered *drifted* when it was introduced at turn T, returned
``True`` immediately after turn T, but returns ``False`` at a later
evaluation point.  The check callable receives the live
:class:`~src.services.sessions.Session` object **after** the turn has been
fully processed (history appended, metadata updated, anchors evaluated,
summary potentially compressed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.services.sessions import Session

# ---------------------------------------------------------------------------
# Canonical N-values
# ---------------------------------------------------------------------------

N_VALUES: tuple[int, ...] = (1, 3, 6, 10, 20)

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class FactAssertion:
    """A single observable fact that should persist from its introduction turn.

    Attributes:
        introduced_at_turn: 1-based index of the turn where this fact was
            stated.  MemoryDriftMetric skips this assertion on earlier turns.
        description: Human-readable label used in metric reports and CSV
            exports.  Should be unique within a profile.
        check: Callable that receives the live ``Session`` object and returns
            ``True`` if the fact is currently retained.  Called after every
            subsequent turn; a ``True → False`` transition is reported as
            drift.
    """

    introduced_at_turn: int
    description: str
    check: Callable["[Session]", bool]


@dataclass
class TurnScript:
    """The scripted input for one conversational turn.

    Attributes:
        turn_index: 1-based position within the scenario (first turn = 1).
        transcript: Raw user message sent to the estimation endpoint.
        facts: Facts *introduced* in this turn that should persist in all
            subsequent turns.  The list may be empty for turns that add no
            new trackable assertions.
    """

    turn_index: int
    transcript: str
    facts: list[FactAssertion] = field(default_factory=list)


@dataclass
class ScenarioProfile:
    """A fully scripted multi-turn conversation profile for stress testing.

    Attributes:
        name: Short machine-readable identifier (snake_case).
        description: One-line summary of the profile's intent.
        turns: Ordered list of up to 20 :class:`TurnScript` objects.
    """

    name: str
    description: str
    turns: list[TurnScript]

    def script_for(self, n_turns: int) -> list[TurnScript]:
        """Return the first *n_turns* turns.

        Args:
            n_turns: Number of turns requested.  Must not exceed
                ``len(self.turns)``.

        Returns:
            A slice ``self.turns[:n_turns]``.

        Raises:
            ValueError: When *n_turns* exceeds the available turns.
        """
        if n_turns > len(self.turns):
            raise ValueError(
                f"Profile '{self.name}' has {len(self.turns)} turns "
                f"(requested {n_turns})."
            )
        return self.turns[:n_turns]

    def all_facts_up_to(self, n_turns: int) -> list[FactAssertion]:
        """Return every :class:`FactAssertion` introduced within the first *n_turns*.

        This is the list MemoryDriftMetric should evaluate after the last
        turn of a run to determine which facts are still retained.

        Args:
            n_turns: Inclusive upper bound on ``FactAssertion.introduced_at_turn``.

        Returns:
            Flat list ordered by introduction turn.
        """
        result: list[FactAssertion] = []
        for turn in self.script_for(n_turns):
            result.extend(turn.facts)
        return result


# ---------------------------------------------------------------------------
# Shared check helpers
# ---------------------------------------------------------------------------


def _has_tech(tech: str) -> Callable["[Session]", bool]:
    """Return a check that passes when *tech* appears in mentioned_technologies."""

    def _check(session: "Session") -> bool:
        return any(
            t.lower() == tech.lower()
            for t in session.metadata.mentioned_technologies
        )

    _check.__name__ = f"has_tech_{tech}"
    return _check


def _project_name_set() -> Callable["[Session]", bool]:
    def _check(session: "Session") -> bool:
        return session.metadata.project_name is not None

    _check.__name__ = "project_name_set"
    return _check


def _project_name_contains(fragment: str) -> Callable["[Session]", bool]:
    def _check(session: "Session") -> bool:
        name = session.metadata.project_name or ""
        return fragment.lower() in name.lower()

    _check.__name__ = f"project_name_contains_{fragment}"
    return _check


def _team_size_is(size: int) -> Callable["[Session]", bool]:
    def _check(session: "Session") -> bool:
        return session.metadata.assumed_team_size == size

    _check.__name__ = f"team_size_is_{size}"
    return _check


def _tier_is_one_of(*tiers: str) -> Callable["[Session]", bool]:
    def _check(session: "Session") -> bool:
        return session.last_resolved_tier in tiers

    _check.__name__ = f"tier_in_{'_'.join(tiers)}"
    return _check


def _scope_contains(fragment: str) -> Callable["[Session]", bool]:
    """Pass when *fragment* appears in agreed_scope OR accumulated_summary."""

    def _check(session: "Session") -> bool:
        scope = session.metadata.agreed_scope or ""
        summary = session.accumulated_summary or ""
        return fragment.lower() in scope.lower() or fragment.lower() in summary.lower()

    _check.__name__ = f"scope_contains_{fragment}"
    return _check


def _anchor_exists(prefix: str) -> Callable["[Session]", bool]:
    """Pass when at least one anchor starts with *prefix* (e.g. 'project_name:')."""

    def _check(session: "Session") -> bool:
        return any(a.startswith(prefix) for a in session.anchors)

    _check.__name__ = f"anchor_exists_{prefix}"
    return _check


# ---------------------------------------------------------------------------
# Profile 1 — GROWING_PROJECT
# ---------------------------------------------------------------------------
#
# Project: "MedScheduler" — a multi-tenant medical appointment platform.
# The stack (FastAPI + PostgreSQL + React) is fixed from turn 1; each
# subsequent turn adds a coherent, incremental feature.
#
# Key assertions:
#  • project_name survives unchanged to turn 20 (never overwritten once set).
#  • Core technologies (FastAPI, PostgreSQL, React) persist in
#    mentioned_technologies throughout all 20 turns.
#  • Tier escalates from "starter" → "standard" → "enterprise" as more
#    expensive features accumulate.
#  • An anchor for project_name is established once the name stabilises.

GROWING_PROJECT = ScenarioProfile(
    name="growing_project",
    description=(
        "Coherent requirements accumulate turn by turn; verifies cost-curve "
        "escalation and project_name persistence at turn 20."
    ),
    turns=[
        TurnScript(
            turn_index=1,
            transcript=(
                "Queremos construir la plataforma MedScheduler para gestión de citas "
                "médicas online. El paciente busca médicos por especialidad, reserva "
                "cita y recibe confirmación por email. Backend con FastAPI y "
                "PostgreSQL, frontend en React. El equipo disponible es de 4 "
                "ingenieros a jornada completa."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=1,
                    description="project_name is set after turn 1",
                    check=_project_name_set(),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="project_name contains 'MedScheduler'",
                    check=_project_name_contains("MedScheduler"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="FastAPI in mentioned_technologies",
                    check=_has_tech("FastAPI"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="PostgreSQL in mentioned_technologies",
                    check=_has_tech("PostgreSQL"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="React in mentioned_technologies",
                    check=_has_tech("React"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="team size is 4",
                    check=_team_size_is(4),
                ),
            ],
        ),
        TurnScript(
            turn_index=2,
            transcript=(
                "Añadir autenticación completa: registro, login con JWT y "
                "OAuth2 (Google y GitHub). Los médicos también deben poder "
                "gestionar su disponibilidad semanal desde un panel propio."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=2,
                    description="OAuth in mentioned_technologies after auth turn",
                    check=_has_tech("OAuth"),
                ),
            ],
        ),
        TurnScript(
            turn_index=3,
            transcript=(
                "La plataforma MedScheduler debe ser multi-tenant: cada clínica "
                "opera bajo su propio subdominio (clinica.medscheduler.com) con datos "
                "completamente aislados a nivel de base de datos. PostgreSQL con "
                "row-level security."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=4,
            transcript=(
                "Añadir un audit log completo: cada acción queda registrada con "
                "timestamp ISO-8601, IP de origen, usuario y descripción de la acción. "
                "El log debe ser inmutable y consultable por el administrador de la clínica."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=5,
            transcript=(
                "Añadir exportación de datos en CSV: historial de citas por médico, "
                "por paciente y por rango de fechas. También exportación de facturación "
                "mensual por clínica en formato Excel."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=6,
            transcript=(
                "Integrar notificaciones por email con SendGrid y SMS con Twilio. "
                "Los pacientes reciben recordatorio 24 h y 1 h antes de la cita. "
                "Las clínicas reciben resumen diario de agenda."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=6,
                    description="SendGrid in mentioned_technologies after notifications turn",
                    check=_has_tech("SendGrid"),
                ),
                FactAssertion(
                    introduced_at_turn=6,
                    description="Twilio in mentioned_technologies after notifications turn",
                    check=_has_tech("Twilio"),
                ),
            ],
        ),
        TurnScript(
            turn_index=7,
            transcript=(
                "Añadir rate limiting en la API pública: 100 peticiones/minuto por "
                "token de acceso. Throttling adicional en el endpoint de búsqueda de "
                "médicos para evitar scraping. Implementar con Redis y un middleware "
                "de FastAPI."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=7,
                    description="Redis in mentioned_technologies after rate-limiting turn",
                    check=_has_tech("Redis"),
                ),
            ],
        ),
        TurnScript(
            turn_index=8,
            transcript=(
                "Construir el panel de administración para superadmins: gestión de "
                "clínicas, activación/desactivación de médicos, estadísticas de uso "
                "globales y facturación por clínica. Frontend en React con gráficas "
                "de Recharts."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=9,
            transcript=(
                "Exponer webhooks para integraciones externas: cuando se crea, cancela "
                "o modifica una cita se envía un evento JSON a la URL configurada por "
                "la clínica. Reintentos automáticos con back-off exponencial."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=10,
            transcript=(
                "Añadir pagos online con Stripe: el paciente puede pagar la consulta "
                "al momento de reservar. Las clínicas reciben el pago con un 2 % de "
                "comisión de plataforma retenida por MedScheduler."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=10,
                    description="Stripe in mentioned_technologies after payments turn",
                    check=_has_tech("Stripe"),
                ),
                FactAssertion(
                    introduced_at_turn=10,
                    description=(
                        "FastAPI still in mentioned_technologies at turn 10 "
                        "(core tech must not drift)"
                    ),
                    check=_has_tech("FastAPI"),
                ),
                FactAssertion(
                    introduced_at_turn=10,
                    description=(
                        "PostgreSQL still in mentioned_technologies at turn 10"
                    ),
                    check=_has_tech("PostgreSQL"),
                ),
            ],
        ),
        TurnScript(
            turn_index=11,
            transcript=(
                "Internacionalizar la plataforma MedScheduler: soporte completo para "
                "español, inglés y francés. Detección automática de idioma del navegador "
                "y selector manual en el perfil de usuario."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=12,
            transcript=(
                "Añadir búsqueda avanzada con Elasticsearch: búsqueda de médicos por "
                "especialidad, nombre, valoración, distancia geográfica y disponibilidad "
                "en tiempo real. Autocompletado con debounce."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=12,
                    description="Elasticsearch in mentioned_technologies after search turn",
                    check=_has_tech("Elasticsearch"),
                ),
            ],
        ),
        TurnScript(
            turn_index=13,
            transcript=(
                "Sistema de valoraciones: tras cada cita el paciente recibe un email "
                "para valorar al médico (1-5 estrellas + comentario libre). Las "
                "valoraciones son verificadas y moderadas antes de publicarse."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=14,
            transcript=(
                "Añadir módulo de telemedicina: videollamadas entre médico y paciente "
                "directamente en la plataforma MedScheduler usando WebRTC. Sala de "
                "espera virtual con estado en tiempo real."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=15,
            transcript=(
                "Exponer una API GraphQL adicional, orientada a clientes mobile y a "
                "integraciones de terceros. La API REST existente se mantiene; GraphQL "
                "es una capa adicional sobre el mismo dominio."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=15,
                    description="GraphQL in mentioned_technologies after API-layer turn",
                    check=_has_tech("GraphQL"),
                ),
            ],
        ),
        TurnScript(
            turn_index=16,
            transcript=(
                "Configurar pipeline CI/CD completo con GitHub Actions: lint, tests "
                "unitarios e integración, build de imagen Docker y despliegue "
                "automático a Kubernetes en GCP. Entornos separados para staging y "
                "producción."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=16,
                    description="Docker in mentioned_technologies after CI/CD turn",
                    check=_has_tech("Docker"),
                ),
                FactAssertion(
                    introduced_at_turn=16,
                    description="Kubernetes in mentioned_technologies after CI/CD turn",
                    check=_has_tech("Kubernetes"),
                ),
                FactAssertion(
                    introduced_at_turn=16,
                    description="GCP in mentioned_technologies after CI/CD turn",
                    check=_has_tech("GCP"),
                ),
            ],
        ),
        TurnScript(
            turn_index=17,
            transcript=(
                "Añadir analíticas en tiempo real para administradores de clínica: "
                "tasa de cancelación, ingresos mensuales, médico con más reservas, "
                "horas pico. Dashboards con gráficas interactivas y alertas "
                "configurables por umbral."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=18,
            transcript=(
                "Integrar con el sistema de seguros médicos: validación automática de "
                "la cobertura del paciente antes de confirmar la cita. Conectores para "
                "tres aseguradoras mediante sus APIs REST propietarias."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=19,
            transcript=(
                "Cumplimiento normativo HIPAA y GDPR: cifrado AES-256 en reposo, "
                "TLS 1.3 en tránsito, gestión de consentimientos explícitos, derecho "
                "al olvido, y contratos DPA con todos los subprocesadores."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=20,
            transcript=(
                "El proyecto MedScheduler está prácticamente completo. Necesitamos "
                "incluir en la estimación: plan de migración de datos históricos desde "
                "el sistema legado, soporte post-lanzamiento de 6 meses y un programa "
                "de formación para el equipo de soporte de cada clínica."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "project_name still set at turn 20 (never overwritten)"
                    ),
                    check=_project_name_set(),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "project_name still contains 'MedScheduler' at turn 20"
                    ),
                    check=_project_name_contains("MedScheduler"),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "FastAPI still in mentioned_technologies at turn 20 "
                        "(no core-tech drift over 20 turns)"
                    ),
                    check=_has_tech("FastAPI"),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description="PostgreSQL still in mentioned_technologies at turn 20",
                    check=_has_tech("PostgreSQL"),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description="React still in mentioned_technologies at turn 20",
                    check=_has_tech("React"),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "tier is standard or enterprise at turn 20 "
                        "(cost must have escalated past starter)"
                    ),
                    check=_tier_is_one_of("standard", "enterprise"),
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Profile 2 — PIVOTING_PROJECT
# ---------------------------------------------------------------------------
#
# Project: "ShopStream" — an e-commerce app.
# Turns 1-4: React (web) + Node.js/Express + MongoDB stack.
# Turn 5: hard pivot — React and Node.js are replaced by Flutter (all
#         platforms) and FastAPI (Python).  Turns 6-10 develop exclusively
#         within the new stack.
#
# Key assertions:
#  • After turn 1: React and MongoDB detected.
#  • After turn 5: Flutter and FastAPI detected (pivot registered).
#  • After turn 5: React STILL in mentioned_technologies (accumulator never
#    removes items) — this is the expected "stale-tech" drift to measure.
#  • After turn 10: Flutter and FastAPI persist (pivot did not fade).
#  • project_name stays "ShopStream" throughout.

PIVOTING_PROJECT = ScenarioProfile(
    name="pivoting_project",
    description=(
        "Stack pivots from React/Node.js to Flutter/FastAPI at turn 5; "
        "verifies pivot detection and surfaces stale-technology accumulation."
    ),
    turns=[
        TurnScript(
            turn_index=1,
            transcript=(
                "ShopStream es una app de e-commerce para pequeños comercios. "
                "Frontend en React, API REST con Node.js y Express, base de datos "
                "MongoDB. El equipo es de 3 personas. Primera versión: catálogo de "
                "productos, carrito y checkout básico."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=1,
                    description="project_name contains 'ShopStream'",
                    check=_project_name_contains("ShopStream"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="React in mentioned_technologies (pre-pivot)",
                    check=_has_tech("React"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="MongoDB in mentioned_technologies (pre-pivot)",
                    check=_has_tech("MongoDB"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="Express in mentioned_technologies (pre-pivot)",
                    check=_has_tech("Express"),
                ),
            ],
        ),
        TurnScript(
            turn_index=2,
            transcript=(
                "Añadir pasarela de pago con Stripe: el comprador introduce la tarjeta "
                "en el checkout de React, el servidor Node.js gestiona el PaymentIntent "
                "y confirma el pedido. También gestión de inventario básica."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=2,
                    description="Stripe in mentioned_technologies (pre-pivot)",
                    check=_has_tech("Stripe"),
                ),
            ],
        ),
        TurnScript(
            turn_index=3,
            transcript=(
                "Sistema de reviews de productos: el comprador puede valorar (1-5 "
                "estrellas) y dejar un comentario. Las reviews se moderan antes de "
                "publicarse. Motor de recomendaciones básico basado en historial de "
                "compras del usuario."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=4,
            transcript=(
                "Añadir app mobile con React Native para iOS y Android, compartiendo "
                "la mayor parte del código de negocio con la web React. Notificaciones "
                "push con Firebase Cloud Messaging."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=4,
                    description="React Native in mentioned_technologies (pre-pivot)",
                    check=_has_tech("React Native"),
                ),
                FactAssertion(
                    introduced_at_turn=4,
                    description="Firebase in mentioned_technologies (pre-pivot)",
                    check=_has_tech("Firebase"),
                ),
            ],
        ),
        TurnScript(
            turn_index=5,
            transcript=(
                "CAMBIO DE ESTRATEGIA: abandonamos React y React Native por completo. "
                "La nueva decisión tecnológica es Flutter para todas las plataformas "
                "(iOS, Android y web). El backend Node.js/Express se reemplaza por "
                "FastAPI en Python, que se integra mejor con los modelos de ML que "
                "añadiremos después. MongoDB se mantiene. Por favor, recalcula la "
                "estimación con el nuevo stack: Flutter + FastAPI + MongoDB."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=5,
                    description=(
                        "Flutter in mentioned_technologies after pivot "
                        "(new stack detected)"
                    ),
                    check=_has_tech("Flutter"),
                ),
                FactAssertion(
                    introduced_at_turn=5,
                    description=(
                        "FastAPI in mentioned_technologies after pivot "
                        "(new backend detected)"
                    ),
                    check=_has_tech("FastAPI"),
                ),
                FactAssertion(
                    introduced_at_turn=5,
                    description=(
                        "React STILL in mentioned_technologies after pivot "
                        "(expected stale-tech accumulation — not a failure, "
                        "but a known drift to surface)"
                    ),
                    # The accumulator never removes items, so React persists.
                    # MemoryDriftMetric should flag this as known drift.
                    check=_has_tech("React"),
                ),
            ],
        ),
        TurnScript(
            turn_index=6,
            transcript=(
                "Continuar con Flutter: implementar pantalla de detalle de producto, "
                "carrito nativo con animaciones, y pantalla de checkout conectada al "
                "nuevo backend FastAPI mediante Dio."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=7,
            transcript=(
                "Implementar notificaciones push en Flutter usando Firebase Cloud "
                "Messaging. Las notificaciones incluyen: confirmación de pedido, "
                "cambio de estado del envío y promociones personalizadas."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=8,
            transcript=(
                "Añadir modo offline en la app Flutter: el catálogo de productos y el "
                "carrito deben funcionar sin conexión usando Hive como base de datos "
                "local. Sincronización automática al recuperar conectividad."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=9,
            transcript=(
                "Configurar autenticación en el nuevo stack: JWT generado por FastAPI, "
                "almacenado de forma segura en Flutter con flutter_secure_storage. "
                "Login social con Google y Apple usando OAuth."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=9,
                    description=(
                        "Flutter still in mentioned_technologies at turn 9 "
                        "(pivot must not fade)"
                    ),
                    check=_has_tech("Flutter"),
                ),
                FactAssertion(
                    introduced_at_turn=9,
                    description=(
                        "FastAPI still in mentioned_technologies at turn 9 "
                        "(pivot must not fade)"
                    ),
                    check=_has_tech("FastAPI"),
                ),
            ],
        ),
        TurnScript(
            turn_index=10,
            transcript=(
                "Añadir autenticación biométrica en Flutter: Touch ID y Face ID para "
                "iOS, huella dactilar para Android. Usar el paquete local_auth. "
                "También necesitamos un panel de administración web ligero con "
                "FastAPI + Jinja2 para gestionar el catálogo de productos."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=10,
                    description=(
                        "Flutter still in mentioned_technologies at turn 10"
                    ),
                    check=_has_tech("Flutter"),
                ),
                FactAssertion(
                    introduced_at_turn=10,
                    description=(
                        "FastAPI still in mentioned_technologies at turn 10"
                    ),
                    check=_has_tech("FastAPI"),
                ),
            ],
        ),
        TurnScript(
            turn_index=11,
            transcript=(
                "Añadir motor de recomendaciones con scikit-learn: modelo de "
                "collaborative filtering entrenado sobre el historial de compras. "
                "El modelo se sirve como microservicio FastAPI y se llama desde la "
                "app Flutter en la pantalla principal."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=11,
                    description="scikit-learn in mentioned_technologies",
                    check=_has_tech("scikit-learn"),
                ),
            ],
        ),
        TurnScript(
            turn_index=12,
            transcript=(
                "Internacionalizar la app Flutter: soporte para español, inglés y "
                "portugués usando el paquete flutter_localizations. Detección "
                "automática según el locale del dispositivo."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=13,
            transcript=(
                "Añadir analytics de comportamiento de usuario: eventos de Flutter "
                "enviados a un pipeline Kafka → BigQuery. Dashboard en Looker Studio "
                "para el equipo de producto."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=13,
                    description="Kafka in mentioned_technologies",
                    check=_has_tech("Kafka"),
                ),
                FactAssertion(
                    introduced_at_turn=13,
                    description="BigQuery in mentioned_technologies",
                    check=_has_tech("BigQuery"),
                ),
            ],
        ),
        TurnScript(
            turn_index=14,
            transcript=(
                "Configurar despliegue con Docker y Kubernetes en AWS. El backend "
                "FastAPI se despliega como servicio en EKS; la base de datos MongoDB "
                "como Atlas gestionado. Pipeline CI/CD con GitHub Actions."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=14,
                    description="Docker in mentioned_technologies",
                    check=_has_tech("Docker"),
                ),
                FactAssertion(
                    introduced_at_turn=14,
                    description="Kubernetes in mentioned_technologies",
                    check=_has_tech("Kubernetes"),
                ),
            ],
        ),
        TurnScript(
            turn_index=15,
            transcript=(
                "Añadir sistema de cupones y descuentos: el administrador crea códigos "
                "de descuento con fecha de expiración, uso máximo y restricciones por "
                "categoría. Los cupones se aplican en el checkout de Flutter."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=16,
            transcript=(
                "Integrar con el sistema de logística de dos operadores de transporte "
                "mediante sus APIs REST. El comprador puede seguir el envío en tiempo "
                "real desde la app Flutter. Notificaciones de estado automáticas."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=17,
            transcript=(
                "Añadir módulo de devoluciones: el comprador solicita la devolución "
                "desde Flutter, el vendedor aprueba o rechaza, se genera la etiqueta "
                "de envío automáticamente y se procesa el reembolso con Stripe."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=18,
            transcript=(
                "Implementar pruebas de carga con Locust antes del lanzamiento: "
                "simular 1.000 usuarios concurrentes comprando simultáneamente. "
                "Identificar cuellos de botella en FastAPI y ajustar los pools de "
                "conexiones a MongoDB."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=19,
            transcript=(
                "Añadir soporte para múltiples divisas: el precio se muestra en la "
                "divisa local del usuario con tipo de cambio en tiempo real. Stripe "
                "gestiona la conversión en el cobro."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=20,
            transcript=(
                "ShopStream está listo para producción. Incluir en la estimación "
                "final: hardening de seguridad (penetration testing), plan de "
                "monitorización con Grafana y Prometheus, y soporte de operaciones "
                "durante los primeros 3 meses post-lanzamiento."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "Flutter still in mentioned_technologies at turn 20 "
                        "(pivot must endure to end)"
                    ),
                    check=_has_tech("Flutter"),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "FastAPI still in mentioned_technologies at turn 20"
                    ),
                    check=_has_tech("FastAPI"),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "React STILL in mentioned_technologies at turn 20 "
                        "(stale-tech accumulation confirmed)"
                    ),
                    check=_has_tech("React"),
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Profile 3 — CONTRADICTING_PROJECT
# ---------------------------------------------------------------------------
#
# Project: "DataVault" — a B2B analytics platform.
# Turn 3 introduces a 30 k€ budget ceiling.
# Turn 8 replaces it with an 80 k€ budget, expanding the scope.
#
# Key assertions:
#  • After turn 3: tier is starter (30 k€ scope → cost below 18 k$ threshold)
#    or standard (if LLM estimates higher despite the stated constraint).
#  • After turn 8: tier should be standard or enterprise (80 k€ budget
#    unlocks larger scope).
#  • After turn 3: the 30k budget appears in agreed_scope or summary.
#  • After turn 8: the 80k budget appears in agreed_scope.
#  • After turn 8: if sliding-window compression has fired, both budgets
#    may coexist in accumulated_summary — surfacing the contradiction.
#  • project_name anchor is established once "DataVault" stabilises.

CONTRADICTING_PROJECT = ScenarioProfile(
    name="contradicting_project",
    description=(
        "Budget stated as 30 k€ at turn 3, revised to 80 k€ at turn 8; "
        "verifies tier resolver updates and contradiction preservation in summary."
    ),
    turns=[
        TurnScript(
            turn_index=1,
            transcript=(
                "DataVault es una plataforma B2B para gestión y análisis de datos "
                "empresariales. Los usuarios necesitan un dashboard de KPIs, "
                "gestión de usuarios por rol y exportación de reportes en PDF. "
                "El stack es Python con FastAPI en el backend y React en el frontend."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=1,
                    description="project_name contains 'DataVault'",
                    check=_project_name_contains("DataVault"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="FastAPI in mentioned_technologies",
                    check=_has_tech("FastAPI"),
                ),
                FactAssertion(
                    introduced_at_turn=1,
                    description="React in mentioned_technologies",
                    check=_has_tech("React"),
                ),
            ],
        ),
        TurnScript(
            turn_index=2,
            transcript=(
                "El equipo de desarrollo disponible es de 2 ingenieros a jornada "
                "completa. La base de datos será PostgreSQL. Necesitamos autenticación "
                "con roles: admin, analista y viewer. Los viewers sólo leen; los "
                "analistas pueden crear reportes; los admins gestionan usuarios."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=2,
                    description="team size is 2",
                    check=_team_size_is(2),
                ),
                FactAssertion(
                    introduced_at_turn=2,
                    description="PostgreSQL in mentioned_technologies",
                    check=_has_tech("PostgreSQL"),
                ),
            ],
        ),
        TurnScript(
            turn_index=3,
            transcript=(
                "Importante: el cliente ha confirmado que el presupuesto máximo "
                "disponible para este proyecto es de 30.000 €. Necesitamos ajustar "
                "el alcance a ese límite. Prioridad: dashboard de KPIs y gestión de "
                "usuarios. Dejar fuera todo lo que no sea esencial."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=3,
                    description=(
                        "30k budget mentioned in scope after turn 3 "
                        "(original constraint)"
                    ),
                    check=_scope_contains("30"),
                ),
                FactAssertion(
                    introduced_at_turn=3,
                    description=(
                        "tier is starter or standard after 30k-budget turn"
                    ),
                    check=_tier_is_one_of("starter", "standard"),
                ),
            ],
        ),
        TurnScript(
            turn_index=4,
            transcript=(
                "Dentro del presupuesto de 30k€, incluir autenticación SSO con SAML "
                "para que las empresas clientes puedan integrar DataVault con sus "
                "proveedores de identidad corporativos (Okta, Azure AD)."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=5,
            transcript=(
                "Añadir un módulo de alertas automáticas: cuando un KPI supera o baja "
                "del umbral configurado, se envía notificación por email y Slack. "
                "Los umbrales los configura el analista desde el dashboard React."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=6,
            transcript=(
                "Integrar con tres fuentes de datos externas: Google Analytics, "
                "Salesforce y HubSpot. Los datos se sincronizan cada hora mediante "
                "conectores REST. El equipo de 2 ingenieros tendrá que gestionar "
                "estas integraciones además del core."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=7,
            transcript=(
                "El cliente quiere explorar capacidades de machine learning para "
                "predecir churn: identificar qué clientes tienen más probabilidad "
                "de abandonar en los próximos 30 días. ¿Es factible dentro del "
                "presupuesto de 30k€ o necesitaríamos más?"
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=7,
                    description=(
                        "30k budget still visible in scope at turn 7 "
                        "(constraint must persist while not revised)"
                    ),
                    check=_scope_contains("30"),
                ),
            ],
        ),
        TurnScript(
            turn_index=8,
            transcript=(
                "El cliente ha revisado internamente el proyecto y ha AMPLIADO el "
                "presupuesto a 80.000 €. Con este nuevo presupuesto, añadir: pipeline "
                "de ML completo con MLflow para tracking de experimentos, un feature "
                "store y el modelo de churn con scikit-learn. También ampliar el "
                "equipo a 4 ingenieros."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=8,
                    description=(
                        "80k budget appears in scope after turn 8 "
                        "(revised constraint)"
                    ),
                    check=_scope_contains("80"),
                ),
                FactAssertion(
                    introduced_at_turn=8,
                    description=(
                        "tier is standard or enterprise after 80k-budget revision "
                        "(larger scope should escalate tier)"
                    ),
                    check=_tier_is_one_of("standard", "enterprise"),
                ),
                FactAssertion(
                    introduced_at_turn=8,
                    description="scikit-learn in mentioned_technologies after ML turn",
                    check=_has_tech("scikit-learn"),
                ),
                FactAssertion(
                    introduced_at_turn=8,
                    description="team size updated to 4 after expansion",
                    check=_team_size_is(4),
                ),
            ],
        ),
        TurnScript(
            turn_index=9,
            transcript=(
                "Con el presupuesto de 80.000 € añadir también un módulo de "
                "seguridad avanzado: penetration testing, cumplimiento SOC 2 Type II, "
                "cifrado AES-256 en reposo para todos los datos de clientes, y "
                "gestión de claves con AWS KMS."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=9,
                    description="AWS in mentioned_technologies after security turn",
                    check=_has_tech("AWS"),
                ),
            ],
        ),
        TurnScript(
            turn_index=10,
            transcript=(
                "El cliente confirma el alcance definitivo con presupuesto de 80.000 €. "
                "Necesitamos un plan de proyecto detallado con hitos mensuales, "
                "criterios de aceptación por fase y un plan de riesgos. DataVault "
                "debe estar en producción en 9 meses."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=10,
                    description=(
                        "80k budget still in scope at turn 10 "
                        "(revised constraint must persist)"
                    ),
                    check=_scope_contains("80"),
                ),
                FactAssertion(
                    introduced_at_turn=10,
                    description=(
                        "tier is standard or enterprise at turn 10"
                    ),
                    check=_tier_is_one_of("standard", "enterprise"),
                ),
                FactAssertion(
                    introduced_at_turn=10,
                    description=(
                        "project_name anchor established for DataVault "
                        "(stable name across turns)"
                    ),
                    check=_anchor_exists("project_name:"),
                ),
            ],
        ),
        TurnScript(
            turn_index=11,
            transcript=(
                "Añadir un módulo de colaboración: los analistas pueden compartir "
                "reportes con anotaciones, comentarios en tiempo real y control de "
                "versiones de los dashboards. Similar a Google Docs pero para datos."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=12,
            transcript=(
                "Integrar Kafka como bus de eventos para que los datos de las tres "
                "fuentes externas lleguen en tiempo real en lugar de sincronización "
                "horaria. Usar Faust (Python) como consumer framework."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=12,
                    description="Kafka in mentioned_technologies",
                    check=_has_tech("Kafka"),
                ),
            ],
        ),
        TurnScript(
            turn_index=13,
            transcript=(
                "Añadir funcionalidad de embedding semántico para búsqueda en "
                "reportes: el analista puede buscar en lenguaje natural entre todos "
                "los reportes históricos. Usar OpenAI embeddings y pgvector en "
                "PostgreSQL para el índice vectorial."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=13,
                    description="OpenAI in mentioned_technologies",
                    check=_has_tech("OpenAI"),
                ),
            ],
        ),
        TurnScript(
            turn_index=14,
            transcript=(
                "Configurar infraestructura en AWS con Terraform: ECS Fargate para "
                "el backend FastAPI, RDS PostgreSQL, ElastiCache Redis y CloudFront "
                "para el frontend React. Todo definido como código."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=14,
                    description="Terraform in mentioned_technologies",
                    check=_has_tech("Terraform"),
                ),
                FactAssertion(
                    introduced_at_turn=14,
                    description="Redis in mentioned_technologies",
                    check=_has_tech("Redis"),
                ),
            ],
        ),
        TurnScript(
            turn_index=15,
            transcript=(
                "Añadir panel de auditoría para cumplimiento: registro inmutable de "
                "todas las consultas de datos con quién, qué, cuándo y desde qué IP. "
                "Exportable en formato JSON y CSV para auditorías externas."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=16,
            transcript=(
                "Implementar multi-tenancy: cada empresa cliente tiene su propio "
                "workspace con datos completamente aislados, cuotas de uso y factura "
                "mensual generada automáticamente según el consumo de API."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=17,
            transcript=(
                "Añadir SDK de Python y JavaScript para que los clientes puedan "
                "enviar sus propios datos a DataVault programáticamente, sin necesidad "
                "de usar los conectores predefinidos."
            ),
            facts=[],
        ),
        TurnScript(
            turn_index=18,
            transcript=(
                "Integrar con dbt para transformaciones de datos: los analistas "
                "pueden definir modelos dbt desde la UI de DataVault y ejecutarlos "
                "sobre sus datos en tiempo real o con schedule."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=18,
                    description="dbt in mentioned_technologies",
                    check=_has_tech("dbt"),
                ),
            ],
        ),
        TurnScript(
            turn_index=19,
            transcript=(
                "Añadir soporte para Snowflake como destino de datos adicional a "
                "PostgreSQL. Los clientes enterprise pueden elegir si sus datos "
                "se almacenan en PostgreSQL gestionado o en su propio Snowflake."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=19,
                    description="Snowflake in mentioned_technologies",
                    check=_has_tech("Snowflake"),
                ),
            ],
        ),
        TurnScript(
            turn_index=20,
            transcript=(
                "DataVault v1.0 completo. El presupuesto final confirmado es de "
                "80.000 €. Incluir en la estimación: soporte SLA 99.9 % con on-call "
                "24/7, programa de onboarding para los primeros 10 clientes enterprise "
                "y un roadmap de 12 meses post-lanzamiento."
            ),
            facts=[
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "80k budget still in scope at turn 20 "
                        "(final confirmed budget persists)"
                    ),
                    check=_scope_contains("80"),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "tier is standard or enterprise at turn 20"
                    ),
                    check=_tier_is_one_of("standard", "enterprise"),
                ),
                FactAssertion(
                    introduced_at_turn=20,
                    description=(
                        "project_name anchor exists at turn 20 "
                        "(DataVault name anchored as stable fact)"
                    ),
                    check=_anchor_exists("project_name:"),
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

ALL_PROFILES: list[ScenarioProfile] = [
    GROWING_PROJECT,
    PIVOTING_PROJECT,
    CONTRADICTING_PROJECT,
]
