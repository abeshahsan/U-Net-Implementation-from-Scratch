

class History():
    def __init__(self):


        self.__metrics: list[str] = ["accuracy"]

        self.train_losses: list = []
        self.train_scores: dict[list] = {}

        self.val_losses: list = []
        self.val_scores: dict[list] = {}

    @property
    def metrics(self):
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics):
        if not isinstance(metrics, list):
            raise TypeError("'metrics' must be a list of string")
        for m in metrics:
            if not isinstance(m, str):
                raise TypeError("'metrics' must be a list of string")

        self.__metrics = metrics

        for metric in self.__metrics:
            self.train_scores[metric] = []
            self.val_scores[metric] = []
            # self.train_losses.

    def set_score_types(self, score_types: list):
        if not isinstance(score_types, list):
            raise TypeError("give the score types as a 'list'")
        self.__metrics = score_types

    def get_scores(self, score_type: dict):
        if not self.__metrics.__contains__(score_type):
            raise KeyError(f"'{score_type}' was not found in the list")


        if self.__val_scores.__contains__([score_type]):
            return [x for x in self.__val_scores]

    def clear(self):
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_scores.clear()
        self.val_scores.clear()

if __name__ == "__main__":
    d = [
    {
        "accuracy": 1,
        "mIoU": 2
    },
    {
        "accuracy": 3,
        "mIoU": 4
    }
    ]

    x = [x["Accuracy"] for x in d if x.__contains__("Accuracy")]

    h = History()
    h.metrics = ["mIoU"]
    print(h.metrics)