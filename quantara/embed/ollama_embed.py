import ollama
from typing import List


class OllamaEmbedding:
    DEFAULT_MODELS = [
        "snowflake-arctic-embed:335m",
        "granite-embedding:30m",
    ]

    def __init__(self, model: str | None = None):
        self.models: List[str] = self.DEFAULT_MODELS.copy()

        if model is not None and model not in self.models:
            self.models.append(model)

        self._verify_models(self.models)
        self._pull_models(self.models)

        self.model = model or self.DEFAULT_MODELS[0]

    def _verify_models(self, models: List[str]) -> None:
        """
        Very basic validation.
        """
        for model in models:
            model_name = model.lower()

            if not any(
                keyword in model_name
                for keyword in (
                    "embed",
                    "embedding",
                    "arctic",
                    "bge",
                    "nomic",
                    "e5",
                )
            ):
                raise ValueError(
                    f"'{model}' does not appear to be an embedding model."
                )

    def _model_exists(self, model: str) -> bool:
        try:
            installed_models = ollama.list().models

            for installed in installed_models:
                if installed.model == model:
                    return True

            return False

        except Exception as e:
            raise RuntimeError(
                f"Failed to check installed Ollama models: {e}"
            ) from e

    def _pull_models(self, models: List[str]) -> None:
        for model in models:
            if not self._model_exists(model):
                print(f"Pulling {model}...")
                ollama.pull(model)
                print(f"Finished downloading {model}")

    def embed(
        self,
        text: str,
        return_only_embedding: bool = True,
    ):
        response = ollama.embed(
            model=self.model,
            input=text,
        )

        if return_only_embedding:
            return response["embeddings"][0]

        return response

    def batch_embed(
        self,
        texts: List[str],
        return_only_embeddings: bool = True,
    ):
        response = ollama.embed(
            model=self.model,
            input=texts,
        )

        if return_only_embeddings:
            return response["embeddings"]

        return response

    def set_model(self, model: str) -> None:
        if not self._model_exists(model):
            print(f"Pulling {model}...")
            ollama.pull(model)

        self.model = model

    def available_models(self) -> List[str]:
        return self.models