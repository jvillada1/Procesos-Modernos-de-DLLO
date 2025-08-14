from groq import Groq

import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DevOpsAgent:
    def __init__(self, answer, api_key):
        self.answer = answer
        self.api_key = api_key

    def create_plan(self):
        client = Groq(api_key=self.api_key)

        prompt = f"""
        Basado en la siguiente información sobre el juego Minecraft:
        "{self.answer}"

        Diseña un plan completo de DevOps que incluya:
        - Estrategia de integración y despliegue continuo (CI/CD).
        - Automatización de pruebas.
        - Estrategia de monitoreo y logging.
        - Escalabilidad y rendimiento.
        - Seguridad y gestión de secretos.

        y coloca (¡) cada vez que interactues
        """

        logger.info("Generando plan DevOps basado en la respuesta dada.")

        response = client.chat.completions.create(
            model="compound-beta",
            messages=[
                {"role": "system", "content": "Eres un ingeniero DevOps experto en videojuegos que diseña estrategias prácticas y escalables."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.5,
            top_p=0.95,
            stream=False,
        )

        return {
            "devops_plan": response.choices[0].message.content
        }