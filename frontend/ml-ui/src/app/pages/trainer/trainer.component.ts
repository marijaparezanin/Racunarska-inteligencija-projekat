import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MlApiService } from '../../services/ml-api.service';


interface TrainingResult {
  accuracy?: number;
  r2_score?: number;
  mae?: number;
  best_params?: any;
  bar_plot_path?: string;
  log_loss_plot_path?: string;
  training_validation_loss_path?: string;
  actual_vs_pred_path?: string;
  error?: string;
  duration: string;
  classification_report: any;
  [key: string]: any;

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
  clfModels = ['Random Forest', 'K Nearest Neighbors', 'Gradient Boosting'];
  databases = ['Diabetes Indicators', 'Diabetes Prediction'];

  classLabels: string[] = ['0', '1', 'macro avg', 'weighted avg'];


  // Selected values
  model_type = this.clfModels[0];
  selectedDatabaseClf = this.databases[0]; // For classification
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
  };

  constructor(private mlApi: MlApiService) {}

  async train() {
    this.resetStatus();
    this.isTraining = true;
    this.trainingMessage = `Training ${this.model_type} on ${this.selectedDatabaseClf}... Please wait.`;
    const payload = {
      model: this.model_mapping[this.model_type],
      dataset: this.selectedDatabaseClf,  // Use classification dataset
    };
    await this.sendTrainingRequest(payload, `${this.model_type} on ${this.selectedDatabaseClf}`);
  }


  private async sendTrainingRequest(payload: any, label: string) {
    try {
      const data = await this.mlApi.trainModel(payload).toPromise();
      this.trainingResult = data as TrainingResult;
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
    window.open(url, '_blank');
  }

  getFilename(path: string): string {
    return path.split('/').pop() || path;
  }

}
