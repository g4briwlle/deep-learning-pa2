## Treinando o modelo

O seguinte comando faz o fluxo padrão de treinamento do modelo final implementado na parte 2, colocando os outputs no caminho `outputs/part2/train_track_b.txt`. Rode qualquer comando no diretório raíz do repo.

### Rodando em UNIX

```bash
python -u -m src.part_2.train_track_b | tee outputs/part2/train_track_b.txt
```

### Rodando em PowerShell

```powershell
py -u -m src.part_2.train_track_b | Tee-Object -FilePath outputs/part2/train_track_b.txt
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