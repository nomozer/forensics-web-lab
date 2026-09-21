import { AnalysisResult, validateAnalysisResult } from '@forensics/shared';

export class JsonReportExporter {
  /**
   * Serializes an AnalysisResult to formatted JSON string, verifying schema compliance.
   */
  public static exportToJson(result: AnalysisResult): string {
    if (!validateAnalysisResult(result)) {
      throw new Error('AnalysisResult does not conform to standardized v1.0.0 schema.');
    }
    return JSON.stringify(result, null, 2);
  }

  /**
   * Initiates client-side file download for the JSON report.
   */
  public static downloadJson(result: AnalysisResult, filenamePrefix = 'forensics_report'): void {
    const jsonStr = this.exportToJson(result);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `${filenamePrefix}_${result.imageInfo.sha256.slice(0, 8)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
