from groq import Groq
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TranslationAgent:
    def __init__(self, text, target_lang, api_key):
        self.text = text
        self.target_lang = target_lang
        self.api_key = api_key

    def translate_content(self):
        client = Groq(api_key=self.api_key)

        prompt = f"""
        Traduce el siguiente contenido al idioma "{self.target_lang}".
        Asegúrate de:
        1. Mantener los nombres de objetos, mobs y comandos de Minecraft en su forma oficial en ese idioma.
        2. Respetar el tono original y evitar traducciones literales que suenen raras.
        3. Devolver únicamente el texto traducido sin explicaciones adicionales.

        Texto original:
        {self.text}
        """

        logger.info(f"Traduciendo contenido a {self.target_lang}...")

        response = client.chat.completions.create(
            model="compound-beta",
            messages=[
                {"role": "system", "content": "Eres un traductor especializado en contenido de Minecraft."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3,
        )

        return {
            "translated_text": response.choices[0].message.content.strip()
        }