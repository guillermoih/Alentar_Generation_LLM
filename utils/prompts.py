# -*- coding: utf-8 -*-
user_prompt_default = """A continuación tienes información sobre una etiqueta CIE-10 y ejemplos de diagnósticos asociados:

- Códigos CIE-10 asociados a los ejemplos: {codigos}
- Nombre de la etiqueta: {nombre_codigo}
- Diagnósticos de ejemplo:
{ejemplos}

Tu tarea es generar exactamente {cantidad} diagnósticos clínicos nuevos, distintos de los anteriores,
que correspondan a la etiqueta indicada ({codigos} - {nombre_codigo}).

Devuelve la respuesta como un JSON con esta estructura exacta:

{{
  "diagnosticos": [
    "Diagnóstico A",
    "Diagnóstico B",
    ...
  ]
}}

Debe ser JSON válido, sin ningún texto adicional."""

start_prompt = """A continuación tienes información sobre una etiqueta CIE-10 y ejemplos de diagnósticos asociados:"""

example_prompt = """
- Códigos CIE-10 asociados al ejemplo: {codigo}
- Nombre de la etiqueta: {nombre_codigo}
- Diagnóstico de ejemplo: {ejemplo}"""

end_prompt = """
Tu tarea es generar exactamente {cantidad} diagnósticos clínicos nuevos, distintos de los anteriores,
que correspondan a la etiqueta {codigo_objetivo}-{nombre_codigo_objetivo}.

Devuelve la respuesta como un JSON con esta estructura exacta:

{{
  "caso 1": {{
      "[lista_codigos]":"Diagnostico 1"
  }},
  "caso 2": {{
      "[lista_codigos]":"Diagnostico 2"
  }},
  ...
}}

Debe ser JSON válido, sin ningún texto adicional."""
