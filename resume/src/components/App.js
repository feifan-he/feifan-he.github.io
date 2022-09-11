import '../index.scss'
import {projects} from '../resumeData'
import {Projects} from './Projects'

function App() {
  return (
      <div className="container">
        <Projects projects={projects}></Projects>
      </div>
  );
}

export default App;
