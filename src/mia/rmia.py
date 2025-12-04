import torch
from src.data.multi import MultiDatasets
from src.mia.base_mia import BaseMIA
from src.mia.shadow_manager import ShadowManager
from src.utils.configs import TrainConfigs, AuditingDataConfigs, ShadowDataConfigs, ModelConfigs
from src.utils.log import SmartLogger, DumbLogger
from typing import Union


class RMIA(BaseMIA):
    def __init__(self, 
                victim_model: torch.nn.Module,
                victim_dataset: MultiDatasets,
                audit_configs: AuditingDataConfigs,
                shadow_configs: ShadowDataConfigs, 
                model_configs: ModelConfigs,
                logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(victim_model=victim_model, victim_dataset=victim_dataset, audit_configs=audit_configs, logger=logger)
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('RMIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(original_datasets=self.victim_dataset,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=shadow_configs)
        self.logger.print_it('RMIA attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=shadow_configs.n_shadow_datasets,
                                                model_configs=model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('RMIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
    #     self.optimize_hyperparameter()

    # def measure_effectiveness(self):
    #     audit_dataset = self.audit_manager.get(labels='mia')
    #     scores=[]
    #     for sample in audit_dataset:
    #         target_prob=prob_target[i]
    #         target_prob_given_target_model=get_prob(target_model,target_data[i:i+1],target_labels[i:i+1],device)
    #         lr_target=target_prob_given_target_model/(target_prob+1e-15)
    #         C=0
    #         for j in range(random_data.size(0)):
    #             rand_prob=rand_probs_per_target[i][j]
    #             rand_prob_given_target_model=get_prob(target_model,random_data[j:j+1],random_labs[j:j+1],device)
    #             lr_rand=rand_prob_given_target_model/(rand_prob+1e-15)
    #             C+=1 if lr_target/lr_rand>gamma else 0

    #         scores.append(C/random_data.size(0))

    #     return np.array(scores)