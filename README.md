# Treinando o modelo

O seguinte comando faz o fluxo padrão de treinamento do modelo final implementado na parte 2, colocando os outputs no caminho `outputs/part2/train_track_b.txt`. Rode qualquer comando no diretório raíz do repo.

### Rodando em UNIX

```bash
python -u -m src.part_2.train_track_b | tee outputs/part2/train_track_b.txt
```

### Rodando em PowerShell

```powershell
py -u -m src.part_2.train_track_b | Tee-Object -FilePath outputs\part2\train_track_b.txt
```

### Rodando com argumentos

Também é possível adicionar argumentos para personalizar o treinamento:

Treino mais curto, só para testar o pipeline
```bash
python -u -m src.part_2.train_track_b --epochs 3 --batch-size 16
```

Trocar o vídeo e o learning rate
```bash
python -u -m src.part_2.train_track_b --video-index 2 --lr 5e-4
```

Salvar em outro arquivo sem sobrescrever o checkpoint da apresentação
```bash
python -u -m src.part_2.train_track_b --epochs 30 --ckpt-name appearance_rnn_30ep.pth
```


# Rodando inferencia.py:
Comando para rodar o arquivo em powershell, com uma sequência de teste, e gerar o video classificado:

```powershell
python inferencia.py --seq_name "MOT17-02-FRCNN" --output "resultado_exemplo.mp4"
```

OBS: Se quiser rodar mais rápidamente, adicionar após a linha 80 do arquivo o seguinte comando para diminuir o número de frames:
```python
# Filtra apenas os primeiros 100 quadros para um teste rápido
dets_df = dets_df[dets_df['frame'] <= 100].copy()
## Avaliando o modelo

Rodando sem argumentos usa o setup padrão que criamos: MOT17-11-FRCNN, primeiros 200 frames, IoU 0.3, cos_threshold=0.15, iou_gate=0.0, usando o checkpoint em outputs/part2/appearance_rnn.pth. O gráfico sai em outputs/part2/3_comparacao_PA1_vs_PA2.png.

### Rodando em UNIX

```bash
python -u -m src.part_2.compare_track_b | tee outputs/part2/compare_track_b.txt
```

### Rodando em PowerShell

```powershell
py -u -m src.part_2.compare_track_b | Tee-Object -FilePath outputs\part2\compare_track_b.txt
```

### Rodando com argumentos

Comparar em outra sequência, com mais frames
```python
python -u -m src.part_2.compare_track_b --seq-name MOT17/train/MOT17-02-FRCNN --max-frames 500
```

Testar um limiar de cosseno mais rígido
```python
python -u -m src.part_2.compare_track_b --cos-threshold 0.3
```

Usar um checkpoint alternativo (ex.: o do Eixo 1 da Parte 3)
```python
python -u -m src.part_2.compare_track_b \
    --ckpt outputs/part3/ckpt_lstm_T32_seed0.pth \
    --plot-name comparacao_lstm_T32.png
```