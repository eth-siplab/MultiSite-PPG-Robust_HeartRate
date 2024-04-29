## Robust Heart Rate Detection via Multi-Site Photoplethysmography (EMBC 2024)

Manuel Meier and [Christian Holz](https://www.christianholz.net)<br/>

[Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland

___________


<p align="center">
<img src="figs/teaser.jpg" width="800">
</p>


---

Smartwatches have become popular for monitoring physiological parameters outside clinical settings.
Using reflective photoplethysmography (PPG) sensors, such watches can non-invasively estimate heart rate (HR) in everyday environments and throughout a patient's day. 
However, achieving consistently high accuracy remains challenging, particularly during moments of increased motion or due to varying device placement. 
In this paper, we introduce a novel sensor fusion method for estimating HR  that flexibly combines samples from multiple PPG sensors placed across the patient's body, including wrist, ankle, head, and sternum (chest).
Our method first estimates signal quality across all inputs to dynamically integrate them into a joint and robust PPG signal for HR estimation.
We evaluate our method on a novel dataset of PPG and ECG recordings from 14 participants who engaged in real-world activities outside the laboratory over the course of a whole day.
Our method achieves a mean HR error of 2.4 bpm, which is 46% lower than the mean error of the best-performing single device (4.4 bpm, head).

Citation
----------
If your find our paper or codes useful, please cite our work:

    @inproceedings{jiang2022avatarposer,
      title={AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing},
      author={Jiang, Jiaxi and Streli, Paul and Qiu, Huajian and Fender, Andreas and Laich, Larissa and Snape, Patrick and Holz, Christian},
      booktitle={Proceedings of European Conference on Computer Vision},
      year={2022},
      organization={Springer}
    }

License and Acknowledgement
----------
This project is released under the MIT license.
