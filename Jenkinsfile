pipeline {
  agent { docker { image 'python:3.7.2' } }
  stages {
    stage('build') {
      steps {
        sh 'pip install -e .[extra]
'
      }
    }
    stage('test') {
      steps {
        sh 'pytest tests'
      }   
    }
  }
}
