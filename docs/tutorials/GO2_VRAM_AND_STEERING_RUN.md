# Go2 AMP + steering: checar a VRAM e rodar o treino

Este guia roda o experimento **AMP + steering com todos os gaits** na RTX 5060 Ti (16 GB) sem transbordar memória para a RAM do sistema.

## Diagnóstico (17/11/2026)

| Sinal | Valor | Leitura |
|---|---|---|
| Treino `go2_loc_amp_allgaits_s42` (4096 envs) | Python com 15,1 GB de VRAM + 1,27 GB na RAM; 100% de uso a 44 W; ~10k samples/s | transbordava para a RAM via PCIe (~8 h para 300M samples) |
| VRAM da NVIDIA com o treino parado | 0,7 GB (Claude, VS Code, Chrome abertos) | o Windows já desenha a tela no vídeo integrado (AMD); não há VRAM a liberar |
| Teste com 4096 envs e fallback desativado | `Warp CUDA error 2: out of memory` no primeiro rollout | **a própria simulação precisa de mais de 16 GB** com 4096 envs |

Conclusão: reduzir o número de envs.

## A configuração de 2048 envs

| | Original (Peng) | Esta config |
|---|---|---|
| Envs em paralelo | 4096 | 2048 |
| `steps_per_iter` | 32 | 64 |
| Samples por iteração | 131.072 | 131.072 |
| Minibatches de ator / crítico / discriminador | 16k / 8k / 8k | iguais |
| Atualizações em 300M samples | ~92k | iguais |
| Checkpoints | a cada 100 iterações (~13M samples) | iguais |

Metade dos robôs, cada um andando o dobro do tempo antes de cada atualização: o lote do PPO e do discriminador fica idêntico. Muda só a diversidade dentro do lote (menos robôs, trechos mais longos de cada um) e o horizonte do GAE (~2,1 s em vez de ~1,1 s).

Arquivos:

- `data/agents/amp_task_go2_2048env_agent.yaml`: igual a `amp_task_go2_agent.yaml`, com `steps_per_iter: 64`
- `args/amp_steering_go2_allgaits_2048env_args.txt`: `--num_envs 2048` + env de steering + o agente acima

---

## 1. Manter o fallback desativado

Com o fallback desativado, se a VRAM acabar o treino **dá erro de memória** em vez de ficar lento em silêncio.

1. Botão direito na área de trabalho → **Painel de Controle NVIDIA**.
2. *Configurações 3D → Gerenciar configurações 3D →* aba **Configurações do programa**.
3. Programa: `C:\Users\hgtll\miniconda3\envs\mimickit\python.exe`.
4. **CUDA – Sysmem Fallback Policy** → **Prefer No Sysmem Fallback** → *Aplicar*.

Deixe assim também no treino longo: se a memória crescer no meio do treino, ele para com erro em vez de ficar 2x mais lento sem aviso. Para desfazer, volte para *Driver Default*.

---

## 2. Teste curto (~5 min)

Abra um **PowerShell fora do app do Claude** (o terminal do app morre se o app fechar):

```powershell
cd C:\Users\hgtll\Documents\Repos\masters\MimicKit
conda activate mimickit
```

```powershell
python mimickit/run.py --arg_file args/amp_steering_go2_allgaits_2048env_args.txt --visualize false --logger txt --max_samples 20000000 --rand_seed 42 --out_dir output/test_vram_steer
```

Em **outro** PowerShell, acompanhe a GPU a cada 5 s:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 5
```

O Gerenciador de Tarefas (*Desempenho → GPU 0 NVIDIA*) mostra o mesmo em gráfico: "Memória dedicada" e "Memória compartilhada".

### Como ler o resultado

Olhe depois de ~3 min (os primeiros minutos incluem compilação de kernels e preenchimento de buffers).

| O que acontece | Significado | Próximo passo |
|---|---|---|
| Roda; memória dedicada < ~15 GB; compartilhada ~0,1 GB; potência > ~100 W | cabe na VRAM | treino longo (seção 3) |
| `CUDA out of memory` | não cabe nem com 2048 | ver seção 4 |
| Roda, mas a potência fica perto de 40–50 W com 100% de uso | gargalo de memória | conferir a memória compartilhada; ver seção 4 |

Também vale anotar o `Samples_Per_Second` impresso no terminal: no treino antigo era ~10k.

No `log.txt` da pasta de teste devem aparecer as colunas `Train_Return_Std` e `Test_Return_Std` (novas).

Ao terminar, apague a pasta de teste:

```powershell
Remove-Item -Recurse -Force output\test_vram_steer
```

---

## 3. Treino longo (1 execução)

```powershell
python mimickit/run.py --arg_file args/amp_steering_go2_allgaits_2048env_args.txt --visualize false --logger tb --save_int_models true --rand_seed 42 --max_samples 300000000 --out_dir output/go2_steer_amp_allgaits_2048env_s42
```

- `--save_int_models true` é **obrigatório**: a análise de colapso usa os checkpoints intermediários (`int_models/`).
- Antes de deixar rodando a noite toda: pause o Windows Update (*Configurações → Windows Update → Pausar por 1 semana*) e confira que o PC não suspende (*Sistema → Energia → Suspender: Nunca*).
- Temperatura: até ~80 °C é normal sob carga; acima de ~87 °C o driver reduz o clock.

TensorBoard (outro PowerShell):

```powershell
python -m tensorboard.main --logdir output/go2_steer_amp_allgaits_2048env_s42 --port 6006 --samples_per_plugin scalars=999999
```

Abra http://localhost:6006. Tags úteis:

| Tag | O que mostra |
|---|---|
| `Test_Return` + `Test_Return_Std` | retorno da tarefa (média e desvio entre os episódios de teste, um por robô) |
| `Train_Return` + `Train_Return_Std` | o mesmo nos episódios de treino (com ruído de exploração) |
| `Disc_Reward_Mean` / `Disc_Reward_Std` | recompensa de estilo |
| `Disc_Agent_Logit` vs `Disc_Demo_Logit` | se a política está enganando o discriminador |

Os pontos aparecem a cada 100 iterações. O desvio mede a variação **entre episódios de uma mesma execução** (estado inicial, velocidade e direção sorteadas), não a variação entre seeds.

---

## 4. Se 2048 ainda não couber

Mude um parâmetro por vez e repita o teste da seção 2.

| Opção | Como | Efeito |
|---|---|---|
| Buffer menor do discriminador | `disc_buffer_size: 100000` em `amp_task_go2_2048env_agent.yaml` | economiza algumas centenas de MB; efeito pequeno no aprendizado |
| 1536 envs × 86 passos | `--num_envs 1536` e `steps_per_iter: 86` | mantém ~132k samples por iteração |

---

## 5. Depois do treino: análise offline

```powershell
python tools/gait_analysis/gait_collapse.py --run_dir output/go2_steer_amp_allgaits_2048env_s42
```

```powershell
python tools/gait_analysis/plot_gait_collapse.py --runs output/go2_steer_amp_allgaits_2048env_s42 --out output/figures/gait_collapse_steer_s42
```

```powershell
python tools/gait_analysis/gait_videos.py --run_dir output/go2_steer_amp_allgaits_2048env_s42
```

### O que sai a mais no steering

No env de steering, `gait_collapse.py` detecta sozinho a velocidade comandada (cada robô tem a sua) e grava:

| Saída | O que é |
|---|---|
| `gait_analysis/speed_bins.csv` | por checkpoint e faixa de velocidade pedida: % de cada gait, % fora do manifold, velocidade pedida, velocidade alcançada na direção do alvo, erro |
| `MI_cmd_gait`, `MI_norm` em `metrics.csv` | informação mútua entre faixa de velocidade e gait; `MI_norm` = fração da incerteza do gait explicada pelo comando (0 = mesmo gait em qualquer velocidade, ou seja, colapso; 1 = a velocidade determina o gait) |
| `v_abs_err` em `metrics.csv` | erro médio absoluto de velocidade |
| `gait_by_speed.png` (no plot) | (a) barras de gait por faixa no checkpoint final; (b) velocidade pedida × alcançada por janela, colorida por gait; (c) `MI_norm` ao longo do treino |

Só entram janelas em que o comando ficou constante durante a janela inteira e por 1 s antes dela (`--settle_sec`), para não contar a transição logo após a troca de alvo. Faixas padrão: `--speed_bins 0.6,1.3,2.0,2.6,3.2` (referências: walk/pace ~1,1, trot ~1,9, run ~3,2 m/s).

Como ler: se nas faixas altas aparece run/trot e a velocidade alcançada acompanha a diagonal, o colapso do location vinha da velocidade fixa. Se o gait é o mesmo em todas as faixas e a velocidade alcançada estaciona abaixo do pedido, o colapso é do AMP.

Para analisar mais rápido (pulando checkpoints), use `--every 2` ou `--every 3`.

Comparabilidade: o treino `go2_loc_amp_allgaits_s42` usou 4096 × 32. Location (velocidade fixa) e steering (velocidade sorteada) diferem também no número de envs; registre isso ao comparar.
