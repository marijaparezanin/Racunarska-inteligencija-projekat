import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MlApiService } from '../../services/ml-api.service';

interface TrainingResult {
  model?: string;
  accuracy?: number;
  classification_report: any;
  duration: string;
  [key: string]: any;  // For dynamic plot keys
}

@Component({
  selector: 'app-trainer',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './trainer.component.html',
  styleUrls: ['./trainer.component.css'],
})
export class TrainerComponent {
  // Sidebar options
  clfModels = ['Random Forest', 'K Nearest Neighbors', 'Gradient Boosting', "Multilayer Perceptron (MLP)", "Deep NN"];
  databases = ['Diabetes Indicators'];

  classLabels: string[] = ['0', '1', 'macro avg', 'weighted avg'];

  // Selected values
  model_type = this.clfModels[0];
  selectedDatabaseClf = this.databases[0];

  // Loading and status
  isTraining = false;
  trainingMessage = '';
  trainingResult: TrainingResult | null = null;
  errorMessage = '';

  model_mapping: Record<string, string> = {
    'Random Forest': 'rf',
    'K Nearest Neighbors': 'knn',
    'Decision Tree': 'dt',
    'Gradient Boosting': 'gb',
    'Multilayer Perceptron (MLP)': 'mlp',
    'Feed Forward NN': 'ff',
    'Deep NN': 'dnn'
  };

  constructor(private mlApi: MlApiService) {}

  async train() {
    this.resetStatus();
    this.isTraining = true;
    this.trainingMessage = `Training ${this.model_type} on ${this.selectedDatabaseClf}... Please wait.`;

    const payload = {
      model: this.model_mapping[this.model_type],
      dataset: this.selectedDatabaseClf
    };

    await this.sendTrainingRequest(payload, `${this.model_type} on ${this.selectedDatabaseClf}`);
  }

  private async sendTrainingRequest(payload: any, label: string) {
    try {
      const data = await this.mlApi.trainModel(payload).toPromise();
      this.trainingResult = data as TrainingResult;
      console.log(this.trainingResult)
      this.isTraining = false;
      this.trainingMessage = `Training ${label} completed in ${this.trainingResult.duration} seconds.`;
      this.errorMessage = '';
    } catch (error: any) {
      this.isTraining = false;
      this.errorMessage = error?.error?.error || 'An error occurred during training.';
      this.trainingMessage = '';
      this.trainingResult = null;
    }
  }

  private resetStatus() {
    this.isTraining = false;
    this.trainingMessage = '';
    this.trainingResult = null;
    this.errorMessage = '';
  }

  downloadImage(url: string) {
    fetch(url)
      .then(res => res.blob())
      .then(blob => {
        const blobUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = url.split('/').pop() || 'image.png';
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(blobUrl);
      })
      .catch(err => console.error('Download failed', err));
  }

  getFilename(path: string): string {
    return path.split('/').pop() || path;
  }

  // Dynamically generate plot list based on keys returned by backend
  getPlotKeys(): { label: string; key: string }[] {
    if (!this.trainingResult) return [];

    const plots: { label: string; key: string }[] = [];

    if (this.trainingResult["bar_plot_path"]) plots.push({ label: 'Bar Plot', key: 'bar_plot_path' });
    if (this.trainingResult["training_validation_loss_path"]) plots.push({ label: 'Training vs Validation Loss Plot', key: 'training_validation_loss_path' });
    if (this.trainingResult["log_loss_plot_path"]) plots.push({ label: 'Log Loss Plot', key: 'log_loss_plot_path' });
    if (this.trainingResult["loss_curve_path"]) plots.push({ label: 'Loss Curve', key: 'loss_curve_path' });
    if (this.trainingResult["roc_curve_path"]) plots.push({ label: 'ROC Curve', key: 'roc_curve_path' });
    if (this.trainingResult["conf_matrix_path"]) plots.push({ label: 'Confusion Matrix', key: 'conf_matrix_path' });

    return plots;
  }
}
