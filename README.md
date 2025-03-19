# Singularity-Indicator
An advanced Multi-Neural Network stock/crypto price predictor. (I am not responsible for any financial gains or losses incurred by the users of this tool).

For use in ATAS trading software.

![Screenshot A](Screenshot1.png)

Instructions:
Place 'SingularityTemplate.cts' in 'Documents\ATAS\Chart\Templates'

Place 'SingularityIndicator.dll' in 'Documents\ATAS\Indicators'

Place 'SingularityDOMTemplate.dts' in 'Documents\ATAS\SmartDOM\Templates'

Restart ATAS.

Load the template 'SingularityTemplate' in ATAS interface.

Load the Smart DOM template via the DOM settings (cog icon).

![Screenshot_B](Screenshot2.png)

(The red and blue lines above and below price are labelled with their respective prediction lengths upon inspecting them in ATAS, for example a blue line of length 8, above price, means the length 8 neural network predicts the indicated price, 8 candles in the future.)
