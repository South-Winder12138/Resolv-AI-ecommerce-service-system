import {registerHooks} from 'node:module'
import {readFileSync} from 'node:fs'
import ts from 'typescript'
import {createElement} from 'react'
import {renderToStaticMarkup} from 'react-dom/server'

registerHooks({
  resolve(specifier,context,next){
    if(specifier.startsWith('.') && !/\.[a-z]+$/.test(specifier)){
      for(const suffix of ['.ts','.tsx']){
        const url=new URL(specifier+suffix,context.parentURL)
        try{readFileSync(url);return {url:url.href,shortCircuit:true}}catch{/* Try the other supported extension. */}
      }
    }
    return next(specifier,context)
  },
  load(url,context,next){
    if(url.endsWith('.css'))return {format:'module',source:'export default {}',shortCircuit:true}
    if(/\.tsx?$/.test(url))return {format:'module',source:ts.transpileModule(readFileSync(new URL(url),'utf8'),{compilerOptions:{jsx:ts.JsxEmit.ReactJSX,module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText,shortCircuit:true}
    return next(url,context)
  }
})
export async function renderComponent(path,name,props){
  const module=await import(new URL(path,import.meta.url))
  return renderToStaticMarkup(createElement(module[name],props))
}
