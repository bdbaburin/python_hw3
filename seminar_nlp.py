from typing import List
from sklearn.feature_extraction.text import CountVectorizer as CV1
import math

# Оригинальный CountVectorizer используется для тестов


class CountVectorizer:
    """Преобразование набора текстовых документов в матрицу подсчета токенов

    Основное использование - векторизация текстовых данных

    Attributes
    ----------
    _corpus_basis: List[str]
        Базисные слова, используемые для векторизации
    by_alphabet: bool
        Требуется ли сортировка базисных слов по умолчанию
        По умолчанию принимает значение False
    punctuation : Str
        Знаки препинания, которые игнорируются векторизатором

    Methods
    -------
    puctuation_cleaner(corpus: List[str]) -> List[str]
        Очищает текст в корпусе от знаков препинания (метод класса)
    fit_transform(corpus: List[str]) -> List[List[int]]
        Переводит корпус текстов в векторное представление
    get_feature_names()
        Выводит базис, использованный при токенизации
    """

    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    def __init__(self, by_alphabet: bool = False):
        self._corpus_basis: List[str] = []
        self.by_alphabet = by_alphabet

    @classmethod
    def puctuation_cleaner(cls, corpus: List[str]) -> List[str]:
        """Убирает знаки препинания из корпуса текстов

        Args:
            corpus: List[str])
            Корпус с текстами

        Returns:
            List[str]: Корпус, не содержащий знаки препинания
        """
        corpus_cleaned = []
        for text in corpus:
            text_copied = text
            for ch in cls.punctuation:
                if ch in cls.punctuation:
                    text_copied = "".join(text_copied.split(ch))
            corpus_cleaned.append(text_copied)
        return corpus_cleaned

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """Векторизует корпус текстов

        Args:
            corpus: List[str])
            Корпус с текстами

        Returns:
            List[List[int]]: Список из векторизованных текстов
        """
        cleaned_corpus = self.puctuation_cleaner(corpus)
        mod_corpus = list(map(lambda x: x.lower(), cleaned_corpus))
        basis = []
        for el in mod_corpus:
            for word in el.split():
                if word not in basis:
                    basis.append(word)
        if self.by_alphabet:
            basis.sort()
        self._corpus_basis = basis
        count_mat = []
        for el in mod_corpus:
            corp_dict = {k: 0 for k in basis}
            for word in el.split():
                corp_dict[word] += 1
            count_mat.append(list(corp_dict.values()))

        return count_mat

    def get_feature_names(self):
        """Выводит базис, использованный при токенизации

        Raises:
            ValueError: Возникает при отсутсвии
            предварительной токенизации
        """
        if not self._corpus_basis:
            raise ValueError("Your vectorizer was not fitted!")
        return self._corpus_basis


class TfidfTransformer:
    """Вычисление tf-idf для векторизованных представлений текстов

    Основное использование - вычисления tf-idf для векторизованных текстов

    Methods
    -------
    idf_transform(matrix: List[List[int]]) -> List[List[float]]
        вычисляет inverse document-frequency по
        заданному представлению корпуса в векторном виде
        (статический метод)
    tf_transform(matx: List[List[int]]) -> List[List[float]]
        Вычисляет term frequency по заданному представлению
        корпуса в векторном виде
        (статический метод)
    tfidf_fit_transform(self, matx: List[List[int]]) -> List[List[float]]
        Вычисляет tf - idf по заданному представлению корпуса в векторном виде
    """

    @staticmethod
    def idf_transform(matrix: List[List[int]]) -> List[float]:
        """Функция вычисляет inverse document-frequency
        по заданному представлению корпуса в векторном виде

        Args:
            matx: List[int])
            Корпус c векторизованными текстами

        Returns:
            List[float]:
            Список из векторизованных inverse document-frequency для текстов
        """
        result = []
        total = len(matrix)
        words_count = len(matrix[0])
        for word_num in range(words_count):
            count = 0
            for doc in matrix:
                if doc[word_num] > 0:
                    count += 1
            result.append(round(math.log((total + 1) / (count + 1)), 3) + 1)
        return result

    @staticmethod
    def tf_transform(matx: List[List[int]]) -> List[List[float]]:
        """Функция вычисляет term frequency по заданному
        представлению корпуса в векторном виде

        Args:
            matx: List[int])
            Корпус c векторизованными текстами

        Returns:
            List[List[float]]:
            Список из векторизованных term frequency для текстов
        """

        tfcount = [[round(item / sum(doc), 3) for item in doc] for doc in matx]
        return tfcount

    def tfidf_fit_transform(self, matx: List[List[int]]) -> List[List[float]]:
        """Функция вычисляет tf - idf по заданному представлению
        корпуса в векторном виде

        Args:
            matx: List[int])
            Корпус c векторизованными текстами

        Returns:
            List[List[float]]: Список из векторизованных tf-idf для текстов
        """
        tfres = TfidfTransformer.tf_transform(matx)
        idfres = TfidfTransformer.idf_transform(matx)
        return [[round(k * z, 3) for k, z in zip(tf, idfres)] for tf in tfres]


class TfidfVectorizer(CountVectorizer):
    """Преобразование набора текстовых документов в матрицу tf-idf
     (обработанное векторное представление)

     Основное использование - вычисления tf-idf для корпусов текстов

    Methods
    -------
    fit_transform(corpus: List[str]) -> List[List[float]]
    Преобразовывает набор текстовых документов в матрицу tf-idf
    """

    def __init__(self):
        super().__init__()
        self.transformer = TfidfTransformer()

    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        """Преобразовывает набор текстовых документов в матрицу tf-idf

        Args:
            corpus: List[str])
            Корпус с текстами

        Returns:
            List[List[float]]: Список из векторизованных tf-idf для текстов
        """
        matx = super().fit_transform(corpus)
        return self.transformer.tfidf_fit_transform(matx)


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never, boil pasta again!!!!!",
        "Pasta Pomodoro! Fresh ingredients Parmesan to taste?",
    ]

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)

    assert vectorizer.get_feature_names() == [
        "crock",
        "pot",
        "pasta",
        "never",
        "boil",
        "again",
        "pomodoro",
        "fresh",
        "ingredients",
        "parmesan",
        "to",
        "taste",
    ]

    assert count_matrix == [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ]

    assert TfidfVectorizer().fit_transform(corpus) == [
        [0.201, 0.201, 0.286, 0.201, 0.201, 0.201, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.143, 0.0, 0.0, 0.0, 0.201, 0.201, 0.201, 0.201, 0.201, 0.201],
    ]

    cat_in_the_hat_docs = [
        "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
        "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
        "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
        "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
        "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)",
    ]

    cv = CV1()
    count_vector = cv.fit_transform(cat_in_the_hat_docs)
    assert count_vector.toarray().tolist() == CountVectorizer(
        by_alphabet=True
    ).fit_transform(cat_in_the_hat_docs)
    print("All tests are passed!")
