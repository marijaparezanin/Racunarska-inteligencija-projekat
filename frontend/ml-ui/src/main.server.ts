import { bootstrapApplication } from '@angular/platform-browser';
import { TrainerComponent } from './app/pages/trainer/trainer.component';
import { provideHttpClient } from '@angular/common/http';

const bootstrap = () =>
  bootstrapApplication(TrainerComponent, {
    providers: [provideHttpClient()]
  });

export default bootstrap;
