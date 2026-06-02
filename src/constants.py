from pathlib import Path

REPORT = Path("report") / "IXI654_report.csv"
 

INPUT  = Path("data/smore/coronal/IXI661-HH-2788-T1/IXI661-HH-2788-T1_smore4.nii.gz")
TARGET = Path("data/IXI_test_resampled_averaged_coronal_interpolated/IXI661-HH-2788-T1.nii.gz")



#INPUT  = Path("data/test/Untitled.png")
#TARGET = Path("data/test/Untitled.png")

def main():
    print(TARGET)
    print(INPUT)

if "__main__" == __name__:
    main()