# Gait_Rehabilitation_MA_AMAM2025
This repo contains the codes and materials for the [paper](https://doi.org/10.26083/tuda-7689) published on AMAM2025. The idea of this work is motivated by [DEP-RL](https://github.com/martius-lab/depRL). 

## Abstract 
Exoskeletons are an attractive research direction in robotics since they involve interactions between humans and machines. Exoskeleton applications can be found in various domains, such as strength augmentation, motion assistance, and gait correction for medical use. Although many studies on exoskeleton control have been conducted, several challenges remain. For one, some methods focus solely on the exoskeleton itself and neglect the interaction/cooperation between humans and machines, resulting in the human having to adapt their motion to the exoskeleton. Additionally, controls are often generated using traditional techniques that don't leverage user data. Although these methods can produce satisfactory results, one of the disadvantages is that they are often limited to providing support passively through predefined parameters, thus leading to poor adaptability across scenarios and latency in the motion. These disadvantages can reduce the effectiveness of the wearable or even cause injury during gait correction in medical applications. This paper presents a control strategy for the exoskeleton that optimizes the generation of healthy gait by correcting pathological gait based on the user’s dynamic properties. 

## Experiments
The whole framework consists of Genarative adversarial imitation learning (GAIL) as one of the Inverse reinforcement learning techniques. The simulation environments are build under [SCONE](https://github.com/tgeijten/sconegym/tree/main), a simulator for predictive biomechanical simulation. 
To run the code, run: 
```bash
python train_imitation.py \
    --algo gail --cuda --env_id /*the env in gym*/ \
    --buffer /*your buffer*/ \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0
```
