import { bootstrapApplication } from '@angular/platform-browser';
import { TrainerComponent } from './app/pages/trainer/trainer.component';
import { provideHttpClient } from '@angular/common/http';

bootstrapApplication(TrainerComponent, {
  providers: [provideHttpClient()]
}).catch(err => console.error(err));
