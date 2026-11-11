"""
Módulo 1: Percepción y Preprocesamiento de Lenguaje Natural (NLP) - EcoMind AI.

Cumple con las normativas técnicas del agente inteligente:
- NORM-PER-01: Validación de entrada no vacía ni compuesta solo por símbolos.
- NORM-PER-02: Normalización estándar (minúsculas, eliminación de acentos/diacríticos,
               remoción de signos de puntuación y colapso de espacios múltiples).
- NORM-PER-03: Preservación de sustantivos clave y calificadores de estado del material
               (ej. 'usado', 'sucio', 'vacio', 'limpio', 'pet', 'hdpe', 'alcalina').
"""

import csv
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Conjunto de palabras vacías (stopwords) en español que no alteran el significado
# ambiental ni el estado del residuo. Se excluyen deliberadamente palabras como
# "sin", "con", "no", "usado", "sucio", "limpio", "vacio" para cumplir NORM-PER-03.
SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "en", "por", "para", "sobre", "entre",
    "y", "e", "o", "u", "que", "como", "donde", "cuando",
    "mi", "tu", "su", "mis", "tus", "sus", "este", "esta",
    "estos", "estas", "ese", "esa", "esos", "esas", "aquel",
    "aquella", "lo", "le", "les", "se", "me", "te", "nos"
}

# Modificadores críticos de estado y material que NUNCA deben eliminarse (NORM-PER-03)
CRITICAL_MODIFIERS = {
    "usado", "usada", "usados", "usadas",
    "sucio", "sucia", "sucios", "sucias",
    "limpio", "limpia", "limpios", "limpias",
    "vacio", "vacia", "vacios", "vacias",
    "enjuagado", "enjuagada", "lavado", "lavada",
    "seco", "seca", "humedo", "humeda",
    "roto", "rota", "rotos", "rotas",
    "grasoso", "grasosa", "grasa", "aceite",
    "pet", "hdpe", "pp", "pead", "pvc", "tetrapak", "unicel",
    "alcalina", "litio", "mercurio", "caduco", "caducos",
    "vencido", "vencidos", "con", "sin", "no"
}


class PerceptionModule:
    """
    Módulo de percepción encargado de recibir, validar, limpiar y normalizar
    las descripciones de residuos escritas en lenguaje cotidiano por el usuario.
    """

    def __init__(self, remove_stopwords: bool = True) -> None:
        """
        Inicializa el preprocesador de texto.

        Args:
            remove_stopwords: Si es True, filtra artículos y preposiciones irrelevantes
                              conservando todos los modificadores críticos de estado.
        """
        self.remove_stopwords = remove_stopwords

    @staticmethod
    def validate_input(text: Optional[str]) -> Tuple[bool, str]:
        """
        NORM-PER-01: Valida que la entrada del usuario no sea nula, vacía,
        ni esté compuesta únicamente por espacios o caracteres no alfanuméricos.

        Args:
            text: Texto original ingresado por el usuario.

        Returns:
            Tupla (es_valido, mensaje_error_o_confirmacion).
        """
        if text is None or not isinstance(text, str):
            return False, "Error (NORM-PER-01): La entrada no puede ser nula."

        stripped = text.strip()
        if not stripped:
            return False, "Error (NORM-PER-01): Por favor escribe el nombre o descripción del residuo."

        # Verificar que contenga al menos una palabra con letras (mínimo 2 caracteres alfabéticos)
        alpha_chars = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]", stripped)
        if len(alpha_chars) < 2:
            return (
                False,
                "Error (NORM-PER-01): La descripción debe contener palabras válidas sobre el residuo."
            )

        return True, "Entrada válida."

    @staticmethod
    def strip_accents(text: str) -> str:
        """
        NORM-PER-02: Elimina acentos y signos diacríticos normalizando en forma NFD,
        convirtiendo además 'ñ' -> 'n' para homologar entradas con o sin tilde.
        """
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def normalize_text(self, text: str) -> str:
        """
        NORM-PER-02 y NORM-PER-03: Ejecuta el flujo completo de normalización:
        1. Conversión a minúsculas.
        2. Normalización de acentos y diacríticos.
        3. Eliminación de signos de puntuación y caracteres especiales.
        4. Colapso de espacios múltiples y filtrado selectivo de stopwords.

        Args:
            text: Texto de entrada validado.

        Returns:
            Cadena normalizada lista para el vectorizador TF-IDF.
        """
        # 1. Minúsculas
        cleaned = text.lower().strip()

        # 2. Normalización de acentos y tildes
        cleaned = self.strip_accents(cleaned)

        # 3. Conservar solo letras (a-z), números (0-9) y espacios
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

        # 4. Colapsar espacios múltiples y dividir en tokens
        tokens = cleaned.split()

        # 5. Filtrado de stopwords respetando calificadores críticos (NORM-PER-03)
        if self.remove_stopwords:
            filtered_tokens = [
                tok for tok in tokens
                if (tok not in SPANISH_STOPWORDS) or (tok in CRITICAL_MODIFIERS)
            ]
            # Si todos los tokens fueron filtrados por accidente, conservar los originales
            if filtered_tokens:
                tokens = filtered_tokens

        return " ".join(tokens)

    def perceive(self, raw_text: Optional[str]) -> Dict[str, object]:
        """
        Ejecuta el paso 1 y 2 del Ciclo del Agente Inteligente (Percepción + Procesamiento).

        Args:
            raw_text: Descripción ingresada por el usuario.

        Returns:
            Diccionario estructurado con el estado de validación, texto original,
            texto normalizado y lista de tokens.
        """
        is_valid, message = self.validate_input(raw_text)
        if not is_valid:
            return {
                "valido": False,
                "mensaje": message,
                "texto_original": raw_text,
                "texto_normalizado": "",
                "tokens": []
            }

        normalized = self.normalize_text(raw_text or "")
        return {
            "valido": True,
            "mensaje": message,
            "texto_original": raw_text,
            "texto_normalizado": normalized,
            "tokens": normalized.split()
        }


def load_and_preprocess_dataset(csv_path: Path) -> List[Dict[str, str]]:
    """
    Carga el archivo CSV del dataset inicial y añade el campo 'texto_normalizado'
    generado por el Módulo de Percepción.
    """
    perception = PerceptionModule(remove_stopwords=True)
    processed_rows: List[Dict[str, str]] = []

    with csv_path.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = perception.perceive(row["texto_entrada"])
            row["texto_normalizado"] = str(result["texto_normalizado"])
            processed_rows.append(row)

    return processed_rows


if __name__ == "__main__":
    # Pruebas demostrativas del Módulo de Percepción (Día 1)
    agente_percepcion = PerceptionModule(remove_stopwords=True)

    ejemplos_usuario = [
        "¡Una botella de plástico PET de 600ml vacía!",
        "Cáscara de plátano y corazón de manzana",
        "Papel sanitario usado de baño",
        "Pila alcalina AA descargada",
        "Caja de cartón de pizza llena de grasa",
        "   ???   "  # Caso inválido para probar NORM-PER-01
    ]

    print("=" * 72)
    print("PRUEBA DEL MÓDULO DE PERCEPCIÓN Y PREPROCESAMIENTO NLP - ECOMIND AI")
    print("=" * 72)
    for entrada in ejemplos_usuario:
        resultado = agente_percepcion.perceive(entrada)
        print(f"\nEntrada original  : {entrada!r}")
        print(f"¿Es válida?       : {resultado['valido']} ({resultado['mensaje']})")
        if resultado["valido"]:
            print(f"Texto normalizado : {resultado['texto_normalizado']!r}")
            print(f"Tokens extraídos  : {resultado['tokens']}")

    # Verificación del dataset_v1.csv si existe en ../data/dataset_v1.csv
    ruta_dataset = Path(__file__).resolve().parent.parent / "data" / "dataset_v1.csv"
    if ruta_dataset.exists():
        registros = load_and_preprocess_dataset(ruta_dataset)
        conteo_categorias: Dict[str, int] = {}
        for r in registros:
            cat = r["categoria"]
            conteo_categorias[cat] = conteo_categorias.get(cat, 0) + 1

        print("\n" + "=" * 72)
        print(f"RESUMEN DE DATASET CARGADO: {ruta_dataset.name}")
        print("=" * 72)
        print(f"Total de registros etiquetados: {len(registros)}")
        for categoria, cantidad in conteo_categorias.items():
            print(f" - {categoria:<18}: {cantidad} registros")
