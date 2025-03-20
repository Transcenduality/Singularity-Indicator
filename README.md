# Singularity-Indicator
An advanced Multi-Neural Network stock/crypto price predictor. (I am not responsible for any financial gains or losses incurred by the users of this tool).

For use in ATAS trading software.

![Screenshot A](Screenshot1.png)

Instructions:
Place 'Holodeck.cts' in 'Documents\ATAS\Chart\Templates'

Place 'PriceNeuralNetwork.dll' in 'Documents\ATAS\Indicators'

Place 'SingularityDOMTemplate.dts' in 'Documents\ATAS\SmartDOM\Templates'

Restart ATAS.

Load the template 'Holodeck' in ATAS interface.

Load the Smart DOM template via the DOM settings (cog icon).

(If the prediction lines are not showing, manually add the indicator from the 'Indicators' panel at the top of your chart.)

![Screenshot_B](Screenshot2.png)

(The red and blue lines above and below price are labelled with their respective prediction lengths upon inspecting them in ATAS, for example a blue line of length 8, above price, means the length 8 neural network predicts the indicated price, 8 candles in the future).

---

Troubleshooting ATAS Not Loading on 5m Chart:

If the 5m chart won't load after an ATAS update, check if ATAS switched BTCUSDT to the Binance Spot pair instead of the Futures pair.

Solution:

Check your BTCUSDT pair in ATAS.

Go to the instrument settings and switch to the Binance Futures BTCUSDT pair instead of the Spot pair.

Reload the 5m chart.

If the issue persists, restart ATAS and try again.
