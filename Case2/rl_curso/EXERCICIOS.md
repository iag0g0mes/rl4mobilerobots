# Exercícios — PPO com stable-baselines3

Este workspace é uma cópia completa e funcional do projeto de navegação com RL. **Tudo já
está implementado** — ambiente, função de recompensa, SAC, avaliação e gráficos — exceto
dois blocos marcados em um único arquivo:

```
src/robot_navigation/robot_navigation/scripts/train_ppo.py
```

Procure pelos banners `EXERCÍCIO 1` e `EXERCÍCIO 2`.

---

## Exercício 1 — Instanciar o agente PPO

Complete a chamada de `PPO(...)`, trocando cada `...` pelo valor correto.

- As linhas marcadas `[OK]` já estão preenchidas — não mexa nelas.
- Quase todos os valores já vêm prontos no objeto `params`, criado por `parse_args()`.
  Veja a dataclass `RunningParams` no topo do arquivo para saber o nome de cada campo
  (`params.lr`, `params.n_steps`, ...).
- Dois argumentos **não** vêm de `params`: o ambiente e o diretório do TensorBoard.
  Ambos já existem como variáveis locais algumas linhas acima.
- A política precisa ser uma rede densa (MLP), porque a observação é um vetor
  (lidar reduzido + posição relativa ao objetivo), não uma imagem.

Documentação: <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>

**Atenção:** `n_steps` precisa ser múltiplo de `batch_size`. O script verifica isso em
`parse_args()` e aborta com `ValueError` se não for.

## Exercício 2 — Disparar o treinamento

Complete a chamada de `model.learn(...)` passando:

- o orçamento total de passos **de ambiente** (vem de `params`);
- a lista `callbacks`, criada logo acima (checkpoints + métricas por episódio);
- a barra de progresso ligada.

---

## Como testar

Antes de preencher, o script quebra logo na criação do PPO — é o esperado.

Build (fora do venv):

```bash
colcon build --symlink-install
source install/setup.bash
```

Terminal 1 — simulação:

```bash
ros2 launch robot_gazebo small_house.launch.py gui:=false
```

Terminal 2 — teste rápido (poucos passos, só para ver se roda):

```bash
python -m robot_navigation.scripts.train_ppo \
  --total-timesteps 4096 --n-steps 512 --batch-size 64 \
  --logdir /tmp/ppo_smoke --env-name smallhouse \
  --ros-args --params-file $(ros2 pkg prefix --share robot_navigation)/params/ros.yaml
```

Deu certo se o robô começa a se mover, a barra de progresso aparece e o diretório
`/tmp/ppo_smoke/smallhouse/<timestamp>/` recebe `monitor.csv`, `metrics.csv` e `tb/`.

Treino de verdade: `./scripts/train_ppo.sh` (1.5M passos — leva horas).

## Referência

Se travar, `scripts/train_sac.py` está completo e tem a mesma estrutura
(construtor do algoritmo + `model.learn(...)`), só que para o SAC. Os hiperparâmetros
são diferentes, mas o formato da chamada ajuda.
