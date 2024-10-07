# Knee-cartilage-segmentation
Swin UNETR segmentation with automated geometry filtering for biomechanical modeling of knee joint cartilage
# YouTube tutorial
https://www.youtube.com/watch?v=jeRuuHDBgW0&t=175s

# Data
Please refer to https://data.mendeley.com/datasets/dc832g7j5m/1 to download the models

For OAI MRIs please refer to https://nda.nih.gov/oai/ (please email the website to request the images)

# Paper
Please cite the following paper:

Kakavand, R., Palizi, M., Tahghighi, P. et al. Integration of Swin UNETR and statistical shape modeling for a semi-automated segmentation of the knee and biomechanical modeling of articular cartilage. Sci Rep 14, 2748 (2024). https://doi.org/10.1038/s41598-024-52548-9

# Abstract
Simulation studies, such as finite element (FE) modeling, offer insights into knee joint biomechanics, which may not be achieved through experimental methods without direct involvement of patients. While generic FE models have been used to predict tissue biomechanics, they overlook variations in population-specific geometry, loading, and material properties. In contrast, subject-specific models account for these factors, delivering enhanced predictive precision but requiring significant effort and time for development. This study aimed to facilitate subject-specific knee joint FE modeling by integrating an automated cartilage segmentation algorithm using a 3D Swin UNETR. This algorithm provided initial segmentation of knee cartilage, followed by automated geometry filtering to refine surface roughness and continuity. In addition to the standard metrics of image segmentation performance, such as Dice similarity coefficient (DSC) and Hausdorff distance, the method's effectiveness was also assessed in FE simulation. Nine pairs of knee cartilage FE models, using manual and automated segmentation methods, were developed to compare the predicted stress and strain responses during gait. The automated segmentation achieved high Dice similarity coefficients of 89.4% for femoral and 85.1% for tibial cartilage, with a Hausdorff distance of 2.3 mm between the automated and manual segmentation. Mechanical results including maximum principal stress and strain, fluid pressure, fibril strain, and contact area showed no significant differences between the manual and automated FE models. These findings demonstrate the effectiveness of the proposed automated segmentation method in creating accurate knee joint FE models.
# Conclusion
In conclusion, the integration of Swin UNETR segmentation model and the proposed automated filtering process
demonstrated remarkable effectiveness in the simulation of knee cartilage. By leveraging the strengths of both Swin
UNETR and automated filtering, this method generated appropriate shapes and geometries for FE models. We have
made our automated segmentation models publicly available (https://data.mendeley.com/datasets/dc832g7j5m/1),
aiming to advance biomechanical modeling and medical image segmentation. We hope this tool will enable the
biomedical community to easily develop efficient subject-specific knee joint models for simulation studies.
