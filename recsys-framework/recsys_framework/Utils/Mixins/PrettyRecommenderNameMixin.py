import re


class PrettyRecommenderNameMixin:
    @staticmethod
    def get_pretty_recommender_name(recommender_name: str) -> str:
        recommender_pretty_name = (
            recommender_name
                .replace("Recommender", "")
                .replace("EarlyStopping", "")
                .replace("_", " ")
        )
        recommender_pretty_name = re.sub("CF$", " CF", recommender_pretty_name)
        recommender_pretty_name = re.sub("CBF$", " CBF", recommender_pretty_name)
        recommender_pretty_name = re.sub("SLIM", "SLIM ", recommender_pretty_name)
        recommender_pretty_name = (
            recommender_pretty_name
                .replace("MatrixFactorization", "MF")
                .replace("SLIM", "SLIM ")
                .replace(" Hybrid", "")
                .replace(" Cython", "")
                .replace(" Wrapper", "")
                .replace(" Matlab", "")
        )
        recommender_pretty_name = re.sub(" +", " ", recommender_pretty_name)

        # Added for CFGAN and GuidelineCFGAN
        recommender_pretty_name = (
            recommender_pretty_name
                .replace("Fixed", "G-")
                .replace("Guideline", "G-")
                .replace("_", " ")
                .replace("ITEM BASED", "i")
                .replace("USER BASED", "u")
                .replace(" ENUMERATED", "")
                .replace(" CONDITIONED INVERTED CONCAT", "")
                .replace(" ZERO RECONSTRUCTION AND PARTIAL MASKING", "ZP")  # Order matters between ZP, PM, and ZR.
                .replace(" PARTIAL MASKING", "PM")
                .replace(" ZERO RECONSTRUCTION", "ZR")
                .replace(" NO MASK", "NM")
                .replace(" CODE HYPER PARAMETERS", " Code")
                .replace(" PAPER HYPER PARAMETERS", " Paper")
        )

        return recommender_pretty_name
