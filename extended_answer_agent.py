from groq import Groq

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtendedAnswerAgent:
    def __init__(self, answer, api_key):
        self.answer = answer
        self.api_key = api_key

    def search(self): 
        client= Groq(api_key=self.api_key)

        prompt = f"""Basado en la siguiente respuesta: "{self.answer}", busca información adicional que pueda complementar
        o mejorar la respuesta dada. Proporciona detalles específicos y relevantes que puedan enriquecer la respuesta original.
        """

        logger.info(f"Prompt para extended answer: {self.answer}")

        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en Minecraft que proporciona información detallada y precisa."},
                {"role": "user", "content": prompt},
            ],
            include_domains= ["minecraft.wiki"],
            max_tokens=2000,
            temperature=0.6,
            top_p=0.95,
            stream=False,
        )

        return {
            "answer": response.choices[0].message.content,
            "tools": response.choices[0].message.executed_tools
        }
