# Demo para o orientador: comandos de visualização

Política rodando ao vivo para cada experimento do relatório "Baseline AMP vanilla: colapso de modo no Go2 multi-gait".

Os comandos terminam em `--devices cpu`, para o Mac. Na máquina com GPU NVIDIA, apague essa flag (o padrão é `cuda:0`).

## Antes de começar

Na raiz do repositório, com o ambiente ativo:

```bash
cd MimicKit
```

```bash
conda activate mimickit
```

Pastas de `output/` que precisam estar no Mac:

| Pasta | Arquivos |
|---|---|
| `output/go2_steer_amp_allgaits_2048env_s42/` | 3 `.yaml`, `model.pt`, `int_models/model_0000001500.pt` |
| `output/go2_loc_amp_allgaits_s42/` | 3 `.yaml`, `model.pt` |
| `output/go2_amp_run_only_2048env_s42/` | 3 `.yaml`, `model.pt`, `int_models/model_0000000600.pt` |

Teclas no visualizador:

| Tecla | Efeito |
|---|---|
| ENTER | pausa / continua |
| Setas, Q / E | câmera |
| W / S (segurar) | velocidade pedida sobe / desce (0,6–3,2 m/s), só nos comandos 2 e 3 |
| A / D (segurar) | gira a direção pedida, só nos comandos 2 e 3 |

Feche a janela para trocar de experimento.

---

## 1. Baseline multi-gait com steering, comandos sorteados

**Mensagem:** é o baseline. A velocidade e a direção são sorteadas a cada 4–7 s, como no treino; a seta mostra o comando. A política só usa walk/pace e trot, nunca run.

```bash
python mimickit/run.py --mode test --num_envs 4 --visualize true --engine_config output/go2_steer_amp_allgaits_2048env_s42/engine_config.yaml --env_config output/go2_steer_amp_allgaits_2048env_s42/env_config.yaml --agent_config output/go2_steer_amp_allgaits_2048env_s42/agent_config.yaml --model_file output/go2_steer_amp_allgaits_2048env_s42/model.pt --devices cpu
```

O que mostrar:

- Quando a seta fica longa (velocidade alta), o robô passa a trotar, mas fica para trás: não passa de ~1,9 m/s.
- Nenhuma fase de voo (as quatro patas nunca saem do chão juntas).
- Cerca de 1 em cada 4 resets começa no meio do clip de run; o robô freia logo em seguida (diagnóstico 1).

## 2. Baseline multi-gait, dirigindo pelo teclado

**Mensagem:** é a mesma política, agora com a velocidade controlada por você. É a forma mais direta de mostrar o colapso.

```bash
python mimickit/run.py --mode test --num_envs 1 --visualize true --engine_config output/go2_steer_amp_allgaits_2048env_s42/engine_config.yaml --env_config data/envs/amp_steering_go2_manual_env.yaml --agent_config output/go2_steer_amp_allgaits_2048env_s42/agent_config.yaml --model_file output/go2_steer_amp_allgaits_2048env_s42/model.pt --devices cpu
```

Roteiro:

1. O robô começa em pé, com 1,0 m/s pedidos: walk/pace.
2. Segure W até ~1,8 m/s: aparece o trot.
3. Segure W até 3,2 m/s (o terminal mostra `Command: speed 3.20 m/s`): ele continua no walk/trot a ~1,6–1,9 m/s e **não corre**, mesmo perdendo recompensa de tarefa.
4. A/D para mostrar que ele obedece a direção normalmente.

## 3. Fase de colapso total (iteração 1500)

**Mensagem:** no meio do treino (~197M samples), a política era 100% walk/pace em qualquer velocidade. O trot só surgiu depois de 260M samples.

```bash
python mimickit/run.py --mode test --num_envs 1 --visualize true --engine_config output/go2_steer_amp_allgaits_2048env_s42/engine_config.yaml --env_config data/envs/amp_steering_go2_manual_env.yaml --agent_config output/go2_steer_amp_allgaits_2048env_s42/agent_config.yaml --model_file output/go2_steer_amp_allgaits_2048env_s42/int_models/model_0000001500.pt --devices cpu
```

Roteiro: o mesmo do comando 2. Segure W até 3,2 m/s: aqui nem o trot aparece.

## 4. Location (velocidade fixa de 1 m/s)

**Mensagem:** é o primeiro experimento, com a velocidade fixa. Ele também colapsou (73% walk, 20% parado, 0% run). O steering foi feito para descartar que a velocidade fixa fosse a causa.

```bash
python mimickit/run.py --mode test --num_envs 4 --visualize true --engine_config output/go2_loc_amp_allgaits_s42/engine_config.yaml --env_config output/go2_loc_amp_allgaits_s42/env_config.yaml --agent_config output/go2_loc_amp_allgaits_s42/agent_config.yaml --model_file output/go2_loc_amp_allgaits_s42/model.pt --devices cpu
```

## 5. Teste 3: AMP só com o clip de run

**Mensagem:** treinado só com o clip de run, o AMP aprende a correr no Go2. Então o run é viável e aprendível, e a ausência dele no multi-gait é colapso de modo, e não limitação física.

```bash
python mimickit/run.py --mode test --num_envs 4 --visualize true --engine_config output/go2_amp_run_only_2048env_s42/engine_config.yaml --env_config output/go2_amp_run_only_2048env_s42/env_config.yaml --agent_config output/go2_amp_run_only_2048env_s42/agent_config.yaml --model_file output/go2_amp_run_only_2048env_s42/model.pt --devices cpu
```

O que mostrar:

- Fase de voo curta, mas real: ~7% do tempo com as quatro patas no ar (o MoCap tem 15%).
- ~2,1 m/s, mais rápido que qualquer gait da política multi-gait, sem quedas.
- Não há tarefa: o robô escolhe a direção sozinho.

## 6. Teste 3, fase lenta antes do run (iteração 600)

**Mensagem:** mesmo sem nenhum clip de walk no dataset, o treino passou por um gait lento parecido com walk antes de chegar ao run. Aqui o discriminador nunca recompensa esse gait lento e continua empurrando a política até o run. No multi-gait, o gait lento é recompensado pelos clips de walk, e a política fica presa nele: é o *mode seeking*.

```bash
python mimickit/run.py --mode test --num_envs 4 --visualize true --engine_config output/go2_amp_run_only_2048env_s42/engine_config.yaml --env_config output/go2_amp_run_only_2048env_s42/env_config.yaml --agent_config output/go2_amp_run_only_2048env_s42/agent_config.yaml --model_file output/go2_amp_run_only_2048env_s42/int_models/model_0000000600.pt --devices cpu
```

Sugestão: rode o 6 e depois o 5, para mostrar a passagem do gait lento para o run.

---

## Ordem sugerida na reunião

1. Comando 2 (teclado, política final): o colapso ao vivo.
2. Comando 3 (iteração 1500): o colapso era total no meio do treino.
3. Comandos 6 e 5 (só run): o run é viável, e o gait lento é uma fase pela qual o treino passa.
4. Comando 1 ou 4, se der tempo.

## Se der erro

- `FileNotFoundError` em um `.pt` ou `.yaml`: a pasta de `output/` não foi copiada inteira (ver a tabela acima).
- Janela abre, mas fica lenta: normal na CPU; use `--num_envs 1` em vez de 4.
- As setas mexem o robô em vez da câmera: o código não está atualizado; rode `git pull origin go2-amp-location`.
