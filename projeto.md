# Parte 1 - Baseline por quadro

## **Detecções públicas**

Das detecções públicas disponíveis no MOT17 ficamos dividos entre a FRCNN e a SDP, uma vez que o DPM é bem menos avançado e provavelmente tornaria o trabalho mais desafiador. Pelo que entendemos SDP é uma técnica mais interessante para esse contexto por ser capaz de lidar melhor com os vídeos em que a câmera está muito distante e os pedestres são "pequenos" demais em relação as outras imagens para que o FRCNN consiga detectá-los com bastante acurácia. Mas, principalmente por ser menos computacionalmente denso, escolhemos o FRCNN por ser um bom meio termo entre o detector ideal e o que apresenta menos tempo de uso nas fases subsequentes.