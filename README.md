# 3DAIRA
3D Ambisonics Impulse Response Analyzer

### TODO
- Implementar el filtro inverso al sine sweep de entrada por canal
  - Se necesita fmin, fmax, duración (la fs se saca de la grabación del sweep) 
- Pasaje de formato A a formato B (DONE)
- Correción en frecuencia del pasaje A a B
- Detección de reflexiones con autocorrelacion o threshold caserito
- Investigar el uso de nivel de intensidad para calcular la dirección de la reflexión
  - El metodo instantaneo [41] y por coherencia entre canales aplicando FFT [43].
      -L Tronchin, M Tubertini, A Ventur, and A Farina, "Implementing spherical microphonearray to determine 3D sound propagation in the "Teatro 1763"
      -J. Merimaa and V. Pulkki, "Spatial Impulse Response Rendering," Proc. of the 7 Int. Conference on Digital Audio Effects (DAFx04), 2004.
  
- Investigar en tps viejos como se llevaron a cabo estos procesamientos
  - Fran Roge (https://github.com/FranciscoRogeVallone/Brojo-Software)
  - Mazalay (https://github.com/imazzala/3D-RIRs/blob/main/main_processing.py)

- El front va a ver
  - Cargar un SS
  - Cargar IR
  - Computar (Generar ISS, convolucionar con SS, procesar, plotear)
  - Cargar plano de planta
  - Dejar fija la ventana de integración por el momento (despues la cambiamos)
  - Exportar gráficos
  
