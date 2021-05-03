pipeline {
  agent { docker { image 'nvcr.io/nvidia/pytorch:21.04-py3' } }
  stages {
    stage('build') {
      steps {
        sh 'pip install -e .[extra]'
        sh 'pip install pytest'
      }
    }
    stage('test') {
      steps {
        sh 'pytest tests'
      }   
    }
  }
}
