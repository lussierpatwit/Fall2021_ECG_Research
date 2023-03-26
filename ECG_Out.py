'''
Written by Paul Lussier and Chen-Hsiang Yu (2021)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from ecgTools import ecgTools
ecgTools = ecgTools(250,20)

# ecgIn = ecgTools.getPhysionetData("18184",'nsrdb/',33000,38000)
ecgTools.readFromSensor()
ecgRaw = ecgTools.getRawData('ecgOut.txt')
ecgFFTa = ecgTools.fftFilt(ecgRaw,63)

ecgButter = ecgTools.butterFilt(ecgFFTa,1, 1, 'high')
ecgSma = ecgTools.sma(ecgRaw,3)
ecgEma = ecgTools.ema(ecgRaw,3,2)
# ecgFFTb = ecgTools.fftFilt(ecgButter,63)
pred = ecgTools.predict(ecgTools.reshape(ecgButter),'/home/pi/Desktop/ECG-Analysis/py-spidev-master/models/11_4_longrun/cp.ckpt')
print(pred)

ecgTools.ecgPlot([ecgRaw,ecgFFTa],0,2500,["Raw", "RFFT"],('Classification:',pred))


