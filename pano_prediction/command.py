import numpy as np
import pandas as pd
import pickle
from pp_exec_env.base_command import BaseCommand, Syntax


class PanoPredictionCommand(BaseCommand):
    # Makes prediction PANO by data of ONE sportsman
    syntax = Syntax(
        [],
    )
    use_timewindow = False  # Does not require time window arguments
    idempotent = True  # Does not invalidate cache

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_progress("Start predict_pano command")
        # that is how you get arguments

        # Make your logic here
        df_to_process = df.drop(["datetime", "protocol"], axis=1)
        df_to_process["date_of_birth"] = pd.to_datetime(
            df_to_process["date_of_birth"].apply(lambda x: x.split("T")[0])
        )
        df_to_process["test_date"] = pd.to_datetime(
            df_to_process["test_date"].apply(lambda x: x.split("T")[0])
        )
        df_to_process["ages"] = round(
            (df_to_process["test_date"] - df_to_process["date_of_birth"])
            / np.timedelta64(365, "D"),
            2,
        )
        df_to_process["ЧСС"] = df_to_process["ЧСС"].apply(
            lambda x: float(x.replace("-", "0")) if isinstance(x, str) else float(x)
        )
        df_to_process = df_to_process.drop("date_of_birth", axis=1)
        grouped = pd.DataFrame(
            df_to_process.groupby(["full_name", "test_date"])["ЧСС"].max()
        )
        names_to_drop = [
            name[0] for name in grouped[grouped["ЧСС"] == 0].index.tolist()
        ]
        dates_to_drop = [
            name[1] for name in grouped[grouped["ЧСС"] == 0].index.tolist()
        ]
        df_to_process = df_to_process[
            ~df_to_process.full_name.isin(names_to_drop)
            & ~df_to_process.test_date.isin(dates_to_drop)
            ]

        features_dict = {
            "age": [],
            "test_type": [],
            "max_v": [],
            "mean_v": [],
            "min_v": [],
            "during_time": [],
            "height": [],
            "weight": [],
        }

        cols_to_calc = [
            col
            for col in df_to_process.columns
            if col
               not in [
                   "t",
                   "test_date",
                   "full_name",
                   "height",
                   "weight",
                   "ages",
                   "WR",
                   "v",
                   "test_type",
                   "train_type",
               ]
        ]
        for col in cols_to_calc:
            features_dict["max_" + col] = []
            features_dict["mean_" + col] = []
            features_dict["min_" + col] = []

        person_date_df = df_to_process.iloc[12:]
        person_date_df = person_date_df[
            (person_date_df["VCO2(STPD)"] < person_date_df["VCO2(STPD)"].quantile(0.95))
            & (
                    person_date_df["VCO2(STPD)"]
                    > person_date_df["VCO2(STPD)"].quantile(0.05)
            )
            ]

        if "-" in person_date_df["WR"].values:
            test_type = 0
        else:
            test_type = 1

        if test_type == 1:
            max_v = person_date_df["WR"].astype(np.float32).max()
            mean_v = person_date_df["WR"].astype(np.float32).mean()
            min_v = person_date_df["WR"].astype(np.float32).min()

        else:
            max_v = person_date_df["v"].astype(np.float32).max()
            mean_v = person_date_df["v"].astype(np.float32).mean()
            min_v = person_date_df["v"].astype(np.float32).min()
        person_date_df = person_date_df.drop(["WR", "v"], axis=1)
        during_time = round(
            float(person_date_df["t"].iloc[-1].split(":")[-2])
            + float(person_date_df["t"].iloc[-1].split(":")[-1]) / 60,
            2,
        )
        height = person_date_df["height"].iloc[0]
        weight = person_date_df["weight"].iloc[0]
        age = person_date_df["ages"].iloc[0]

        for col in cols_to_calc:
            #                 print(name, date, col)
            if "-" in person_date_df[col].values:
                #                     print(person_date_df[col])
                person_date_df[col] = person_date_df[col].apply(
                    lambda x: x.replace("-", "0") if isinstance(x, str) else x
                )

            features_dict["max_" + col].append(
                person_date_df[col].astype(np.float32).max()
            )
            features_dict["mean_" + col].append(
                person_date_df[col].astype(np.float32).mean()
            )
            features_dict["min_" + col].append(
                person_date_df[col].astype(np.float32).min()
            )

        # features_dict["name"].append(name)
        # features_dict["test_date"].append(date)
        features_dict["age"].append(age)
        features_dict["test_type"].append(test_type)
        features_dict["max_v"].append(max_v)
        features_dict["min_v"].append(min_v)
        features_dict["mean_v"].append(mean_v)
        features_dict["during_time"].append(during_time)
        features_dict["height"].append(height)
        features_dict["weight"].append(weight)
        # Add description of what going on for log progress
        self.log_progress("First part is complete.", stage=1, total_stages=2)
        #

        # Use ordinary logger if you need

        df_to_predict = pd.DataFrame(features_dict)
        prep_pipeline = pickle.load(
            open(
                "/home/rsv/work/postprocessing/extended_commands/pp_cmd_pano_prediction/"
                "pano_prediction/models/pano_prep_pipeline.pkl",
                "rb",
            )
        )
        predict_data = prep_pipeline.transform(df_to_predict)

        model = pickle.load(
            open(
                "/home/rsv/work/postprocessing/extended_commands/pp_cmd_pano_prediction/"
                "pano_prediction/models/pano_model.pkl",
                "rb",
            )
        )
        predicted_pano = model.predict(predict_data)
        df["pano"] = predicted_pano[0]

        self.log_progress("Prediction is complete", stage=2, total_stages=2)
        return df.reset_index(drop=True)
