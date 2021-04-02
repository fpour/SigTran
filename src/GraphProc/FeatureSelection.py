"""
Specify the inclusion of the features in the experiments
"""
import pandas as pd


class FeatureStatus:
    def __init__(self):
        self.feature_stat = self.generate_feature_status()

    def generate_feature_status(self):
        """"
        generate the status of different features
        """
        feature_df_column_name = ['feature', 'select']
        feature_dict = {
            # major features
            0: ['node', 1],

            1: ['address', 0],  # NEVER select
            2: ['isp', 0],  # NEVER select
            3: ['is_anchor', 0],  # NEVER select

            5: ['degree', 0],
            6: ['in_degree', 1],  # this
            7: ['out_degree', 0],

            8: ['avg_amount_in_tx', 0],
            9: ['min_amount_in_tx', 0],
            10: ['max_amount_in_tx', 0],
            11: ['sum_amount_in_tx', 1],  # this
            12: ['std_amount_in_tx', 1],  # this
            13: ['ent_amount_in_tx', 0],

            14: ['avg_in_tx_interval', 1],  # this
            15: ['min_in_tx_interval', 0],
            16: ['max_in_tx_interval', 1],  # this
            17: ['sum_in_tx_interval', 1],  # this
            18: ['std_in_tx_interval', 0],
            19: ['ent_in_tx_interval', 0],

            20: ['avg_amount_out_tx', 0],
            21: ['min_amount_out_tx', 0],
            22: ['max_amount_out_tx', 0],
            23: ['sum_amount_out_tx', 1],  # this
            24: ['std_amount_out_tx', 0],
            25: ['ent_amount_out_tx', 0],

            26: ['avg_out_tx_interval', 0],
            27: ['min_out_tx_interval', 0],
            28: ['max_out_tx_interval', 0],
            29: ['sum_out_tx_interval', 0],
            30: ['std_out_tx_interval', 0],
            31: ['ent_out_tx_interval', 0],

            32: ['avg_amount_all_tx', 0],
            33: ['min_amount_all_tx', 0],
            34: ['max_amount_all_tx', 0],
            35: ['sum_amount_all_tx', 0],
            36: ['std_amount_all_tx', 0],
            37: ['ent_amount_all_tx', 1],  # this

            38: ['avg_all_tx_interval', 0],
            39: ['min_all_tx_interval', 0],
            40: ['max_all_tx_interval', 0],
            41: ['sum_all_tx_interval', 0],
            42: ['std_all_tx_interval', 0],
            43: ['ent_all_tx_interval', 0],

            44: ['no_edge_within_egonet', 0],
            45: ['no_edge_in_egonet', 1],  # this
            46: ['no_edge_out_egonet', 0],
            47: ['no_edge_all_egonet', 1],  # this

            48: ['avg_neighbor_degree', 0],
            49: ['min_neighbor_degree', 0],
            50: ['max_neighbor_degree', 0],
            51: ['sum_neighbor_degree', 0],
            52: ['std_neighbor_degree', 0],
            53: ['ent_neighbor_degree', 0],

            54: ['avg_neighbor_w_degree', 0],
            55: ['min_neighbor_w_degree', 0],
            56: ['max_neighbor_w_degree', 0],
            57: ['sum_neighbor_w_degree', 0],
            58: ['std_neighbor_w_degree', 0],
            59: ['ent_neighbor_w_degree', 0],

            60: ['avg_neighbor_in_degree', 0],
            61: ['min_neighbor_in_degree', 0],
            62: ['max_neighbor_in_degree', 0],
            63: ['sum_neighbor_in_degree', 0],
            64: ['std_neighbor_in_degree', 0],
            65: ['ent_neighbor_in_degree', 0],

            66: ['avg_neighbor_out_degree', 0],
            67: ['min_neighbor_out_degree', 0],
            68: ['max_neighbor_out_degree', 0],
            69: ['sum_neighbor_out_degree', 0],
            70: ['std_neighbor_out_degree', 0],
            71: ['ent_neighbor_out_degree', 0],

            # 62: ['balance', 1, 0],

            # derived features
        }

        feature_df = pd.DataFrame.from_dict(feature_dict, orient='index',
                                            columns=feature_df_column_name)

        return feature_df


def main():
    """
    check functionality
    """
    feature_st = FeatureStatus()
    print(feature_st.feature_stat.head())


if __name__ == '__main__':
    main()

