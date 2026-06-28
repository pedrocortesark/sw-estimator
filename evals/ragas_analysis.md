# Análisis de métricas RAGAS - Session 11

## Tabla de métricas

| Query | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|-------|--------------|------------------|-------------------|----------------|
| Q1 (Mobile banking) | 0.4762 | 0.4994 | 0.9750 | 0.7857 |
| Q2 (E-commerce) | 0.2857 | 0.3976 | 0.9333 | 0.0000 |
| Q3 (Telemedicine) | 0.4783 | 0.3468 | 1.0000 | 0.3529 |
| Q4 (IoT Factory) | 0.7273 | 0.0000 | 1.0000 | 0.6923 |
| Q5 (Payments) | 0.0000 | 0.4277 | 0.9468 | 0.2500 |
| **PROMEDIO** | **0.3935** | **0.3343** | **0.9710** | **0.4162** |

## Observaciones críticas

**Lo que más chirría:**

1. **Context Recall muy bajo en Q2 (0.0000) y Q3 (0.3529)**: El contexto recuperado no cubre los elementos clave del ground truth. En Q2 (e-commerce), el pipeline no recupera chunks que hablen de "promotional discounts" ni "multi-currency pricing" que sí están en el ground truth. En Q3 (telemedicine), falta cobertura sobre "GDPR-compliant consent management".

2. **Faithfulness bajo en Q5 (0.0000)**: La respuesta generada para el sistema de pagos no es fiel al contexto recuperado. El modelo está inventando cifras o módulos que no están en los chunks recuperados, lo que indica alucinación severa.

3. **Answer Relevancy en 0.0000 para Q4**: Aunque el contexto es preciso (1.0000), la respuesta generada no es relevante a la pregunta original sobre IoT factory monitoring. El modelo se desvía del tema.

## Diagnóstico

El problema principal es que **el pipeline de generación no está aprovechando bien el contexto recuperado**. Aunque Context Precision es excelente (0.9710 promedio), lo que significa que recuperamos chunks relevantes, el modelo:
- No cita correctamente las fuentes (faithfulness bajo)
- Inventa información no presente en el contexto (alucinaciones)
- Se desvía del tema en algunos casos (answer relevancy inconsistente)

Esto sugiere que el **prompt de generación necesita mejoras** para forzar al modelo a:
1. Usar únicamente la información del contexto proporcionado
2. Citar explícitamente las fuentes para cada afirmación
3. Mantenerse enfocado en la pregunta original

## Próximos pasos para el directo

- Implementar detección de alucinaciones post-generación
- Mejorar el prompt para forzar citación explícita
- Añadir validación de que todas las cifras mencionadas estén en el contexto
- Evaluar el impacto de estas mejoras en las métricas RAGAS
