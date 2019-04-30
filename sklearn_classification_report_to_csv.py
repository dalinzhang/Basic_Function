import pandas as pd
from sklearn.metrics import classification_report

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe



if __name__ == "__main__":
	writer = pd.ExcelWriter("/home/dalinzhang/program/19CIKM/result/result.xlsx")
	report = classification_report(y_true = np.argmax(np.array(true_test).reshape([-1, 2]), 1).flatten(), y_pred = np.array(pred_test).flatten())
	report.to_excel(writer, 'report', index=False)
	writer.save()
