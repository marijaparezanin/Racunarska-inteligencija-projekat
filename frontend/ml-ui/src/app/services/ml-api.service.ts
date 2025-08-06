import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class MlApiService {
  private baseUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) {}

  trainModel(payload: any) {
    console.log('Training payload:', payload);
    return this.http.post(`${this.baseUrl}/train`, payload);
  }
}
