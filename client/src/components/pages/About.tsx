import React from 'react';

function About() {
  return (
    <div className="md:max-w-4xl sm:max-w-sm sm:mx-auto top-0 left-0 right-0 my-48 mx-4">
      <div className="text-center text-2xl bg-gradient-to-r from-transparent via-white/30 to-transparent p-1 mb-12">
        About Us
      </div>

      <div className="flex flex-col gap-8">
        <div className="text-center text-xl">
          Money Laundering has increased in recent years with the growth of the
          digital era. As such, the need to have a system to detect this type of
          financial fraud is imperative.{' '}
          <span className="font-bold text-emerald-500">Morgan</span> is a model
          that has been developed to detect money laundering using{' '}
          <span className="font-bold text-rose-500">Neural Networks</span>{' '}
          integrated with <span className="font-bold text-rose-500">XAI</span>.
        </div>

        <div className="text-center text-xl">
          {
            "The XAI method used is Counterfactual Explanations which explains the reasoning behind the model's decision making in human readable explanations. As such, the explanations provided are easily understood by non-technical personnel."
          }
        </div>
      </div>
    </div>
  );
}

export default About;
