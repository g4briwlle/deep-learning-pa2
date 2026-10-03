# Documento de uso de IA generativa ao longo do trabalho

**Nota sobre o fluxo de trabalho:** Devido ao tempo curto para a realização do trabalho e à proximidade com a semana de provas, o uso de ferramentas de Inteligência Artificial (Gemini e DeepSeek) foi significativamente maior neste PA em comparação ao anterior. As IAs atuaram não apenas como ferramentas de pesquisa, mas como verdadeiros *pair programmers*, acelerando a codificação de rotinas *boilerplate*, diagnosticando incompatibilidades matemáticas e ajudando a estruturar a defesa teórica para a apresentação.

## Parte 1: Baseline e Detecções

### Escolha da opção de detecção pública do MOT17
Utilizamos o Gemini para aprofundar no artigo "Tracking Without Bells and Whistles" (que trata dos detectores disponíveis no MOT17) para ajudar na escolha do detector perguntando qual a complexidade computacional e o tempo que deve demorar rodar cada um ao longo do resto do assingment.

## Parte 2: Memória Temporal (Trilha B - Memória de Aparência)

*   **Design da Arquitetura e Extração de Features:** Daniel usou o DeepSeek para ajudar a estruturar a leitura otimizada das imagens do arquivo ZIP e o pipeline de extração de *crops* para a ResNet18.
*   **Debugging de Performance (O grande gargalo):** O nosso modelo inicialmente apresentou um desempenho de *ID Switches* muito pior do que a baseline geométrica da Parte 1. Utilizamos o Gemini para analisar os códigos de treino e inferência. A IA diagnosticou dois problemas conceituais críticos:
    1.  **Incompatibilidade Métrica:** Estávamos treinando com `TripletMarginLoss` (distância Euclidiana por padrão) e fazendo a associação com Similaridade de Cosseno. A IA sugeriu aplicar normalização L2 (`F.normalize`) logo na saída da rede.
    2.  **Backbone Congelado e Efeito "Teletransporte":** A IA explicou que a ResNet18 100% congelada era "cega" para detalhes de roupas, e sugeriu descongelar apenas a `layer4` para focar nas *features* humanas sem causar *overfitting* ou explodir o tempo de treino na GPU local[cite: 9]. Além disso, a IA ajudou a Gabrielle a programar um "portão de raio espacial" para impedir que IDs fossem teletransportados sob oclusão no rastreador.

## Parte 3: Ablação

*   **Automação e Velocidade:** Como o tempo era curto e a GPU estava ocupada, escolhemos o Eixo 3 (o que entra na recorrência: aparência vs. geometria vs. híbrido)[cite: 9]. Gabrielle usou o Gemini para gerar rapidamente a arquitetura baseada apenas em Geometria (usando *Linear layers* em vez da ResNet), o que permitiu rodar o laço das 3 *seeds* exigidas instantaneamente na CPU[cite: 9].
*   A IA também auxiliou na geração do script de plotagem (`matplotlib`) com a média e o desvio padrão e na fundamentação da justificativa teórica sobre o comportamento da rede em multidões densas versus oclusões longas[cite: 9].

## Parte 4: Galeria de Falhas e Horizonte de Memória

*   **Comprovação Analítica:** Daniel pediu ao DeepSeek dicas de como extrair gradientes intermediários no PyTorch. Em seguida, Gabrielle usou o Gemini para escrever o roteiro que faz um *forward* manual de $T=32$ quadros, forçando o arquivamento das normas com `h_t.retain_grad()` para plotar o decaimento exponencial e provar o problema do gradiente que some[cite: 9].
*   **Geração Visual e Debugging:** Usamos IA para criar a função que desenha as caixas delimitadoras lado a lado (Ground Truth vs. Predição). Durante o processo, esbarramos em um erro de extração de dados (`KeyError: 0`). Fornecemos o *Traceback* ao Gemini, que rapidamente identificou que estávamos indexando um dicionário nomeado como se fosse uma lista de tensores, corrigindo a extração do *DataFrame* em segundos.

## Parte 5: Teste de Estresse

*   **Adaptação de Código:** A IA foi fundamental para refatorar o `DetectorSimulator` (da Parte 0/1) que corrompe caixas sintéticas (aplicando falsos positivos e remoções), ensinando-o a ler e corromper o *Ground Truth* real das sequências do MOT17[cite: 9].
*   **Refinamento Analítico:** O Gemini nos ajudou a corrigir uma conclusão teórica inicial incorreta. Em vez de "amplificar" o erro do detector (como a baseline da Parte 1 fazia), a IA nos ajudou a embasar matematicamente como a memória de aparência da Trilha B tem a capacidade de *absorver* falhas e omissões do detector graças à preservação do estado latente[cite: 9].

## Estruturação Final
Utilizamos o Gemini em uma etapa final para compilar todos os artefatos gerados (curvas analíticas, mosaicos visuais e tabelas) em um "Super Resumo", garantindo que a nossa defesa de apresentação oral respondesse a todas as perguntas obrigatórias estipuladas pelo professor ao longo das seções do enunciado[cite: 9].

## Bibliografia
1. Tracking Without Bells and Whistles (Bergmann, Meinhardt e Leal-Taixe) - [arXiv:1903.05625](https://arxiv.org/abs/1903.05625)