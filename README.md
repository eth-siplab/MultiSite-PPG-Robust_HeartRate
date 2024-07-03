## Robust Heart Rate Detection via Multi-Site Photoplethysmography (EMBC 2024)

Manuel Meier and [Christian Holz](https://www.christianholz.net)<br/>

[Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland

---

>Smartwatches have become popular for monitoring physiological parameters outside clinical settings. Using reflective photoplethysmography (PPG) sensors, such watches can noninvasively estimate heart rate (HR) in everyday environments and throughout a patient’s day. However, achieving consistently high accuracy remains challenging, particularly during moments of increased motion or due to varying device placement. In this paper, we introduce a novel sensor fusion method for estimating HR that flexibly combines samples from multiple PPG sensors placed across the patient’s body, including wrist, ankle, head, and sternum (chest). Our method first estimates signal quality across all inputs to dynamically integrate them into a joint and robust PPG signal for HR estimation. We evaluate our method on a novel dataset of PPG and ECG recordings from 14 participants who engaged in real-world activities outside the laboratory over the course of a whole day. Our method achieves a mean HR error of 2.4 bpm, which is 46% lower than the mean error of the best-performing single device (4.4 bpm, head).


![teaser](https://github.com/eth-siplab/MultiSite-PPG-Robust_HeartRate/blob/main/figs/teaser.png)

Code will be released very soon.

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
