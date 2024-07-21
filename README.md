## Robust Heart Rate Detection via Multi-Site Photoplethysmography (EMBC 2024)

Manuel Meier and [Christian Holz](https://www.christianholz.net)<br/>

[Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland

[link to the EMBC paper](tbd)

---

>Smartwatches have become popular for monitoring physiological parameters outside clinical settings. Using reflective photoplethysmography (PPG) sensors, such watches can noninvasively estimate heart rate (HR) in everyday environments and throughout a patient’s day. However, achieving consistently high accuracy remains challenging, particularly during moments of increased motion or due to varying device placement. In this paper, we introduce a novel sensor fusion method for estimating HR that flexibly combines samples from multiple PPG sensors placed across the patient’s body, including wrist, ankle, head, and sternum (chest). Our method first estimates signal quality across all inputs to dynamically integrate them into a joint and robust PPG signal for HR estimation. We evaluate our method on a novel dataset of PPG and ECG recordings from 14 participants who engaged in real-world activities outside the laboratory over the course of a whole day. Our method achieves a mean HR error of 2.4 bpm, which is 46% lower than the mean error of the best-performing single device (4.4 bpm, head).


![teaser](https://github.com/eth-siplab/MultiSite-PPG-Robust_HeartRate/blob/main/figs/teaser.png)

Use
----------
Execute `main.py`. Without any modification, it will run with the provided example data of 1 hour of 4 recording traces at 128 Hz.
HR errors for all contributing PPG traces and the fusion trace are printed.

To use with your own PPG data, modify the corresponding lines in the file header.

Plotting
---------
To minimize dependency issues, plotting functions are commented out by default.
To plot the ppg traces and the resulting trace and/or the generated template waves,
follow these steps:
- remove commenting in the plot sections of the header of both the `main.py` and `ppg_fusion_functions.py` files.
- remove commenting from the plot function of the template class (bottom of `ppg_fusion_functions.py`)
- removing commenting on line 45 of `main.py` for template plotting
- remove commenting at bottom of `main.py` for trace plotting

example of template plot:
![template](https://github.com/eth-siplab/MultiSite-PPG-Robust_HeartRate/blob/main/figs/template_plot.png)



Citation
----------
If your find our paper or codes useful, please cite our work:

    @inproceedings{meier2024robustHR,
      title={Robust Heart Rate Detection via Multi-Site Photoplethysmography},
      author={Meier, Manuel and Holz, Christian},
      booktitle={2024 46th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
      year={2024},
      organization={IEEE}
    }

License and Acknowledgement
----------
This project is released under the MIT license.
